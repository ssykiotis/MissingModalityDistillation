import hydra
import os
import logging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import shutil

from src.utils import *
from src.training.Trainer import *
from torchmetrics.classification import BinaryROC
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image




logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@hydra.main(version_base = "1.3", config_path = "config", config_name = "inference_exp_results_config")
def main(config: DictConfig) -> None:
    if config.device  == 'cuda':
        if not torch.cuda.is_available():
            logging.info('GPU not available: switching to CPU')
            config.device = 'cpu'

    config.device = 'cpu'
    setup_seed(config.seed)

    config['log_location'] = HydraConfig.get().runtime.output_dir
    config.model.n_channels = config['n_channels']


    ds_parser = hydra.utils.instantiate(config.dataset)
    trainer   = Trainer(config, ds_parser)

    #create images folder
    os.mkdir(f'{config.log_location}/images')

    config.model.n_channels = config.n_teacher_channels

    #instantiate models and load weights
    teacher_model  = hydra.utils.instantiate(config.model)
    teacher_model  = teacher_model.to(config.device)
    teacher_model.load_state_dict(torch.load(config.teacher_model_path))

    config.model.n_channels = 2
    
    baseline_model = hydra.utils.instantiate(config.model)
    baseline_model = baseline_model.to(config.device)
    baseline_model.load_state_dict(torch.load(config.baseline_model_path))
    student_model  = hydra.utils.instantiate(config.model)
    student_model  = student_model.to(config.device)
    student_model.load_state_dict(torch.load(config.student_model_path))

    metric = Dice(multiclass = False).to(config.device)
    teacher_roc   = BinaryROC().to(config.device)
    baseline_roc  = BinaryROC().to(config.device)
    kd_roc        = BinaryROC().to(config.device)

    test_dl     = trainer.get_dataloader('test')
    image_names = ds_parser.test_samples

    f1_list_teacher,f1_list_baseline, f1_list_student = [], [], []

    j = 0

    x_min = test_dl.dataset.norm_params.x_min
    x_max = test_dl.dataset.norm_params.x_max

    performance = pd.DataFrame(data = None, columns = ['teacher','baseline','student'])

    for idx,batch in enumerate(test_dl):
        x, y, *x_missing = [item.float().to(config.device) for item in batch]

        y = y[:,0].to(torch.int64)

        _, logits_teacher  = teacher_model(x)
        teacher_roc.update(logits_teacher[:,1],  y)
        y_pred_teacher     = logits_teacher.argmax(axis = 1).detach().cpu().numpy()


        _, logits_baseline = baseline_model(x_missing[0])
        baseline_roc.update(logits_baseline[:,1], y)
        y_pred_baseline    = logits_baseline.argmax(axis = 1).detach().cpu().numpy()

        _, logits_student  = student_model(x_missing[0])
        kd_roc.update(logits_student[:,1], y)
        y_pred_student     = logits_student.argmax(axis = 1).detach().cpu().numpy()


        img_names_batch = image_names[idx*config.batch_size:idx*config.batch_size + config.batch_size]

        # y = y.detach().cpu().numpy()
        x = x.detach().cpu().numpy()


        for i in range(x.shape[0]):
            y = torch.tensor(y).to(config.device)

            y_t = torch.tensor(y_pred_teacher).to(config.device)
            y_b = torch.tensor(y_pred_baseline).to(config.device)
            y_s = torch.tensor(y_pred_student).to(config.device)

            f1_teacher  = metric(y_t[i],y[i])
            f1_baseline = metric(y_b[i],y[i])
            f1_student  = metric(y_s[i],y[i])

            f1_list_teacher.append(f1_teacher.item())
            f1_list_baseline.append(f1_baseline.item())
            f1_list_student.append(f1_student.item())

            y = y.detach().cpu().numpy()

            vv = x[i,0,:,:]
            vh = x[i,1,:,:]

            vv = ((vv-vv.min())/ (vv.max()-vv.min()) * 255)
            vh = ((vh-vh.min())/ (vh.max()-vh.min()) * 255)

            vv,_ = image_histogram_equalization(vv)
            vh,_ = image_histogram_equalization(vh)

            vv = vv.astype('uint8')
            vh = vh.astype('uint8')

            # vv_png = Image.fromarray(vv)
            # vh_png = Image.fromarray(vh)

            # vv_png.save(f'{config.log_location}/images/{j}_vv.png')
            # vh_png.save(f'{config.log_location}/images/{j}_vh.png')




            # #########################Sentinel 1###########################

            # mask_original =  y[i].astype('uint8')
            # R                 = mask_original.copy() * 0
            # G                 = mask_original.copy() * 0
            # B                 = mask_original.copy() * 255
            # mask_original_rgb = np.stack((R,G,B), axis=-1)
            # png_mask_original = Image.fromarray(mask_original_rgb, mode = 'RGB')
            # png_mask_original.save(f'{config.log_location}/images/{j}_mask.png')


            # mask_teacher  = y_pred_teacher[i].astype('uint8')
            # R                = mask_teacher.copy() * 255
            # G                = mask_teacher.copy() * 0
            # B                = mask_teacher.copy() * 0
            # mask_teacher_rgb = np.stack((R,G,B), axis=-1)
            # png_mask_teacher = Image.fromarray(mask_teacher_rgb, mode = 'RGB')
            # png_mask_teacher.save(f'{config.log_location}/images/{j}_teacher.png')


            # mask_baseline = y_pred_baseline[i].astype('uint8')
            # R                 = mask_baseline.copy() * 255
            # G                 = mask_baseline.copy() * 0
            # B                 = mask_baseline.copy() * 0
            # mask_baseline_rgb = np.stack((R,G,B), axis=-1)
            # png_mask_baseline = Image.fromarray(mask_baseline_rgb, mode = 'RGB')
            # png_mask_baseline.save(f'{config.log_location}/images/{j}_baseline.png')


            # mask_student  = y_pred_student[i].astype('uint8')
            # R                = mask_student.copy() * 255
            # G                = mask_student.copy() * 0
            # B                = mask_student.copy() * 0
            # mask_student_rgb = np.stack((R,G,B), axis=-1)
            # png_mask_student = Image.fromarray(mask_student_rgb, mode = 'RGB')
            # png_mask_student.save(f'{config.log_location}/images/{j}_student.png')

            # vv_png = vv_png.convert(mode = 'RGB')
            # vh_png = vh_png.convert(mode = 'RGB')

    
            # vv_teacher = Image.blend(vv_png,png_mask_teacher,0.3)
            # vh_teacher = Image.blend(vh_png,png_mask_teacher,0.3)

            # vv_teacher_gt = Image.blend(vv_teacher,png_mask_original,0.3)
            # vh_teacher_gt = Image.blend(vh_teacher,png_mask_original,0.3)

            # vv_baseline = Image.blend(vv_png,png_mask_baseline,0.3)
            # vh_baseline = Image.blend(vh_png,png_mask_baseline, 0.3)

            # vv_baseline_gt = Image.blend(vv_baseline,png_mask_original,0.3)
            # vh_baseline_gt = Image.blend(vh_baseline,png_mask_original,0.3)

            # vv_student = Image.blend(vv_png,png_mask_student,0.3)
            # vh_student = Image.blend(vv_png,png_mask_student,0.3)

            # vv_student_gt = Image.blend(vv_student,png_mask_original,0.3)
            # vh_student_gt = Image.blend(vh_student,png_mask_original,0.3)

            # vv_teacher.save(f'{config.log_location}/images/{j}_vv_teacher.png')
            # vh_teacher.save(f'{config.log_location}/images/{j}_vh_teacher.png')

            # vv_teacher_gt.save(f'{config.log_location}/images/{j}_vv_teacher_gt.png')
            # vh_teacher_gt.save(f'{config.log_location}/images/{j}_vh_teacher_gt.png')


            # vv_baseline.save(f'{config.log_location}/images/{j}_vv_baseline.png')
            # vh_baseline.save(f'{config.log_location}/images/{j}_vh_baseline.png')

            # vv_baseline_gt.save(f'{config.log_location}/images/{j}_vv_baseline_gt.png')
            # vh_baseline_gt.save(f'{config.log_location}/images/{j}_vh_baseline_gt.png')

            # vv_student.save(f'{config.log_location}/images/{j}_vv_student.png')
            # vh_student.save(f'{config.log_location}/images/{j}_vh_student.png')

            # vv_student_gt.save(f'{config.log_location}/images/{j}_vv_student_gt.png')
            # vh_student_gt.save(f'{config.log_location}/images/{j}_vh_student_gt.png')



            j+=1
    
    logger.info('Plotting RocAUC!')

    # roc_score_teacher  = teacher_roc.compute()
    # roc_score_baseline = baseline_roc.compute()
    # roc_score_student  = kd_roc.compute()

    # logging.info('ROC Teacher:', roc_score_teacher)
    # logging.info('ROC Baseline:',roc_score_baseline)
    # logging.info('ROC Student:',roc_score_student)

    fig, ax = plt.subplots()
    teacher_roc.plot( score = True, ax = ax)
    baseline_roc.plot(score = True, ax = ax)
    kd_roc.plot(      score = True, ax = ax)

    ax.set_title('ROC Curve for the different models')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{config.log_location}/rocauc.png', dpi = 300)

    performance['teacher']  = f1_list_teacher
    performance['baseline'] = f1_list_baseline
    performance['student']  = f1_list_student

    performance.to_csv(f'{config.log_location}/performance.csv')


if __name__=="__main__":
    main()