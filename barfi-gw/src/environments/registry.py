from src.environments.Maze import Maze
from src.environments.Maze_badreward import Maze_badreward
from src.environments.Maze_goodreward import Maze_goodreward
from src.environments.Maze_noreward import Maze_noreward
from src.environments.MountainCarDense import MountainCarDense
from src.environments.MountainCarSparse import MountainCarSparse



def get_environment(name):
    
    if name == 'Maze':
        return Maze
    if name == 'Maze_badreward':
        return Maze_badreward
    if name == 'Maze_goodreward':
        return Maze_goodreward
    if name == 'Maze_noreward':
        return Maze_noreward
    if name == 'MountainCarDense':
        return MountainCarDense

    raise NotImplementedError()

