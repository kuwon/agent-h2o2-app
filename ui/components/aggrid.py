
import streamlit as st
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
    _AGGRID=True
except Exception:
    _AGGRID=False
def aggrid_table(df, height=260, selection="single"):
    if not _AGGRID:
        st.dataframe(df, height=height, use_container_width=True); return {"selected_rows":[]}
    import pandas as pd
    if not hasattr(df,"columns"): df = pd.DataFrame(df)
    gob = GridOptionsBuilder.from_dataframe(df); gob.configure_pagination(enabled=True); gob.configure_selection(selection_mode=selection, use_checkbox=True); gob.configure_grid_options(domLayout="normal"); grid_options=gob.build()
    return AgGrid(df, gridOptions=grid_options, height=height, update_mode=GridUpdateMode.SELECTION_CHANGED, fit_columns_on_grid_load=True, allow_unsafe_jscode=True, theme="alpine")
