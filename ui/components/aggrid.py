# ui/components/aggrid.py
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import AgGrid
import pandas as pd
from typing import Dict, Optional

def aggrid_table(
    df: pd.DataFrame,
    key: str,
    selection_mode: str = "none",           # "none" | "single" | "multiple"
    height: int = 280,
    enable_filter: bool = True,
    fit_columns_on_load: bool = False,
    allow_horizontal_scroll: bool = True,
    display_labels: Optional[Dict[str, str]] = None,
    enable_header_checkbox: bool = False,   # 👈 추가: 헤더에서 전체 선택/해제
):
    if df is None:
        df = pd.DataFrame()

    gob = GridOptionsBuilder.from_dataframe(df, enableRowGroup=True, enableValue=True, enablePivot=True)
    gob.configure_default_column(
        sortable=True, resizable=True, filter=enable_filter,
        wrapText=False, autoHeight=False
    )

    if selection_mode in ("single", "multiple"):
        gob.configure_selection(
            selection_mode=selection_mode,
            use_checkbox=(selection_mode == "multiple"),
            rowMultiSelectWithClick=True,
        )

    # 헤더 체크박스(첫 컬럼에)
    if enable_header_checkbox and selection_mode == "multiple" and len(df.columns) > 0:
        first_col = df.columns[0]
        gob.configure_column(first_col,
            headerCheckboxSelection=True,
            headerCheckboxSelectionFilteredOnly=False,  # 필터 무관 전체 토글
            checkboxSelection=True,
        )

    if display_labels:
        for field, header in display_labels.items():
            if field in df.columns:
                gob.configure_column(field, headerName=header)

    grid_options = gob.build()
    grid_options["suppressHorizontalScroll"] = not bool(allow_horizontal_scroll)

    return AgGrid(
        df,
        gridOptions=grid_options,
        update_on=["filterChanged", "modelUpdated", "selectionChanged"],  # 👈 선택 변경 이벤트
        height=height,
        key=key,
        fit_columns_on_grid_load=bool(fit_columns_on_load),
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False,
    )
