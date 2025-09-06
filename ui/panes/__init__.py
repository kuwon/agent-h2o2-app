from typing import Literal
from .info import render_info_pane
from .sim import render_sim_pane
LeftView=Literal["info","sim"]
def render_left_pane(view: LeftView="info"):
    render_sim_pane() if view=="sim" else render_info_pane()
