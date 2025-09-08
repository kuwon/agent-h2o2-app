from typing import Optional, Dict
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

def _mk_select_all_handler(select_filtered_only: bool) -> JsCode:
    # 선택 방식만 문장으로 분기하고, 본문은 f-string 없이 순수 문자열로 유지
    select_stmt = (
        "api.forEachNodeAfterFilterAndSort(function(node){ node.setSelected(true); });"
        if select_filtered_only else
        "api.selectAll();"
    )
    js = """
function(params) {
  setTimeout(function() {
    var api = params.api;
    //__SELECT__
    api.refreshHeader();
  }, 0);
}
""".replace("//__SELECT__", select_stmt)
    return JsCode(js)

def aggrid_table(
    df: pd.DataFrame,
    key: str,
    selection_mode: str = "none",           # "none" | "single" | "multiple"
    height: int = 280,
    enable_filter: bool = True,
    fit_columns_on_load: bool = False,
    allow_horizontal_scroll: bool = True,
    display_labels: Optional[Dict[str, str]] = None,
    enable_header_checkbox: bool = False,   # 헤더에서 전체 선택/해제
    # 추가 옵션
    select_all_on_load: bool = False,       # 최초 렌더 때 전체 선택
    select_all_on_data_change: bool = False,# rowData 바뀔 때마다 전체 선택
    select_filtered_only: bool = False,     # 필터된 행만 선택
    row_id_field: Optional[str] = None,     # 고유 ID 컬럼(선택)
):
    if df is None:
        df = pd.DataFrame()

    gob = GridOptionsBuilder.from_dataframe(df, enableRowGroup=True, enableValue=True, enablePivot=True)
    gob.configure_default_column(
        sortable=True, resizable=True, filter=enable_filter,
        wrapText=False, autoHeight=False
    )

    # 안정적인 선택/갱신을 위해 getRowId 지정(선택)
    if row_id_field and row_id_field in df.columns:
        gob.configure_grid_options(
            getRowId=JsCode(
                "function(params) {{ return params.data['{}']; }}".format(row_id_field)
            )
        )

    if selection_mode in ("single", "multiple"):
        gob.configure_selection(
            selection_mode=selection_mode,
            use_checkbox=(selection_mode == "multiple"),
            rowMultiSelectWithClick=True,
        )

    # 헤더 체크박스(첫 컬럼)
    if enable_header_checkbox and selection_mode == "multiple" and len(df.columns) > 0:
        first_col = df.columns[0]
        gob.configure_column(
            first_col,
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

    # 이벤트 핸들러 주입 (중복 코드 없이 안전하게)
    handler = _mk_select_all_handler(select_filtered_only)

    if selection_mode == "multiple" and select_all_on_load:
        grid_options["onFirstDataRendered"] = handler

    if selection_mode == "multiple" and select_all_on_data_change:
        grid_options["onRowDataChanged"] = handler
        grid_options["onRowDataUpdated"] = handler

    return AgGrid(
        df,
        gridOptions=grid_options,
        update_on=["filterChanged", "modelUpdated", "selectionChanged"],
        height=height,
        key=key,
        fit_columns_on_grid_load=bool(fit_columns_on_load),
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False,
    )
