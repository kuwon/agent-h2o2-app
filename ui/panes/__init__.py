from .info import render_info_pane
from .sim import render_sim_pane

def render_left_pane(view: str = "info"):
    if view == "sim":
        render_sim_pane()
    else:
        render_info_pane()
