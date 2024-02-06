from dataclasses import dataclass

@dataclass
class NormParams:
	x_min:  int = None
	x_max:  int = None
	x_mean: int = None
	x_std:  int = None