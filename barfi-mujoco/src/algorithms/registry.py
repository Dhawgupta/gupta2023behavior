from src.algorithms.ActorCritic import ActorCritic
from src.algorithms.REINFORCE import REINFORCE
from src.algorithms.REINFORCEPotential import REINFORCEPotential
from src.algorithms.Barfi import Barfi
from src.algorithms.ActorCritic import ActorCritic
from src.algorithms.ActorCriticOnline import ActorCriticOnline
from src.algorithms.Frodo import Frodo
from src.algorithms.ActorCritic2 import ActorCritic2
from src.algorithms.ActorCriticPotential import ActorCriticPotential
from src.algorithms.ActorCritic2Potential import ActorCritic2Potential

def get_agent(name):
    if name == 'ActorCritic':
        return ActorCritic
    if name == 'ActorCriticOnline':
        return ActorCriticOnline
    if name == 'REINFORCE':
        return REINFORCE
    if name == 'Barfi':
        return Barfi
    if name == 'REINFORCEPotential':
        return REINFORCEPotential
    if name == 'Frodo':
        return Frodo
    if name == 'ActorCritic2':
        return ActorCritic2
    if name == 'ActorCriticPotential':
        return ActorCriticPotential
    if name == 'ActorCritic2Potential':
        return ActorCritic2Potential
    
    
    

    raise NotImplementedError()
