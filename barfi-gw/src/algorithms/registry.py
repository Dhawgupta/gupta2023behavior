from src.algorithms.ActorCritic import ActorCritic
from src.algorithms.REINFORCE import REINFORCE
from src.algorithms.REINFORCEPotential import REINFORCEPotential
from src.algorithms.Barfi import Barfi
from src.algorithms.BarfiNeumann import BarfiNeumann
from src.algorithms.ActorCriticPotential import ActorCriticPotential

def get_agent(name):
    if name == 'ActorCritic':
        return ActorCritic
    if name == 'REINFORCE':
        return REINFORCE
    if name == 'Barfi':
        return Barfi
    if name == 'BarfiNeumann':
        return BarfiNeumann
    if name == 'REINFORCEPotential':
        return REINFORCEPotential
    if name == 'ActorCriticPotential':
        return ActorCriticPotential
    

    raise NotImplementedError()
