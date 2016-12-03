import ale_python_interface.ale_python_interface as ale
from scipy import misc

breakout = ale.ALEInterface()
breakout.loadROM('../Arcade-Learning-Environment-0.5.1/roms/Breakout.bin')
breakout.getLegalActionSet()
breakout.getMinimalActionSet()
breakout.getScreen()
"""Outputs 210 x 160 image. You can also pass in an np array if you want it to
fill that for you."""
grays = breakout.getScreenGrayscale()
"""It has shape (210, 160, 1) and we need (210, 160)"""
grays_2d = grays.reshape((210, 160))
downsampled_grays = misc.imresize(grays_2d, (110, 84))
