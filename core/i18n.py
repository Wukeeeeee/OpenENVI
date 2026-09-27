"""OpenENVI Internationalization (i18n) Module.

Provides dynamic runtime bilingual translation between English and Simplified Chinese (简体中文).
All internal keys and source code conventions remain strictly in English.
"""

from typing import Dict, Optional
from PySide6.QtCore import QObject, Signal


TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "en": {
        # App & Window
        "app.title": "OpenENVI - Remote Sensing Platform",
        "app.ready": "Ready",
        "app.initialized": "OpenENVI initialized successfully",
        "app.about_title": "About OpenENVI",
        "app.about_desc": (
            "<h3>OpenENVI Remote Sensing Platform</h3>"
            "<p>A lightweight, open-source ENVI-style remote sensing and hyperspectral image analysis platform.</p>"
            "<p><b>Phase 0: Project Scaffolding & Modular UI Shell</b></p>"
        ),

        # Menu Titles
        "menu.file": "&File",
        "menu.view": "&View",
        "menu.layer": "&Layer",
        "menu.display": "&Display",
        "menu.tools": "&Tools",
        "menu.window": "&Window",
        "menu.language": "&Language",
        "menu.help": "&Help",

        # Menu & Toolbar Actions
        "action.open": "Open...",
        "action.exit": "Exit",
        "action.probe": "Probe",
        "action.pan": "Pan",
        "action.zoom_in": "Zoom In",
        "action.zoom_out": "Zoom Out",
        "action.fit": "Fit",
        "action.reset_layout": "Reset Dock Layout",
        "action.about": "About OpenENVI",
        "action.layer_props": "Layer Properties...",
        "action.remove_layer": "Remove Active Layer",

        # Stretch Modes
        "stretch.label": " Stretch: ",
        "stretch.linear2": "Linear 2%",
        "stretch.linear5": "Linear 5%",
        "stretch.equalize": "Equalization",
        "stretch.gaussian": "Gaussian",
        "stretch.none": "No Stretch",

        # Layer Manager Dock
        "dock.layer_manager": "Layer Manager",
        "layer_manager.col_name": "Layer Name",
        "layer_manager.col_type": "Type",
        "layer_manager.btn_remove": "Remove",
        "layer_manager.btn_up": "Up",
        "layer_manager.btn_down": "Down",
        "layer_manager.msg_removed": "Layer removed",

        # Data Manager Dock
        "dock.data_manager": "Data Manager",
        "data_manager.rb_gray": "Gray Scale",
        "data_manager.rb_rgb": "RGB Color",
        "data_manager.rgb_group": "RGB Assignment",
        "data_manager.btn_load_band": "Load Band",
        "data_manager.btn_load_rgb": "Load RGB",
        "data_manager.btn_close_file": "Close File",
        "data_manager.msg_select_band": "Please select a band first",
        "data_manager.msg_assign_rgb": "Please assign all R, G, and B bands",
        "data_manager.msg_file_closed": "Dataset closed",

        # Toolbox Dock
        "dock.toolbox": "Toolbox",
        "toolbox.search_placeholder": "Filter tools...",
        "toolbox.cat_basic": "Basic Tools",
        "toolbox.cat_transforms": "Transforms",
        "toolbox.cat_spectral": "Spectral",
        "toolbox.cat_classification": "Classification",
        "toolbox.cat_indices": "Indices",

        # Toolbox Tools
        "toolbox.tool_band_math": "Band Math",
        "toolbox.tool_resize": "Resize Data (Spatial/Spectral)",
        "toolbox.tool_stacking": "Layer Stacking",
        "toolbox.tool_mosaic": "Masking & Mosaicking",
        "toolbox.tool_stats": "Raster Statistics",
        "toolbox.tool_roi": "Region of Interest (ROI) Tool",
        "toolbox.tool_pca": "Principal Components Analysis (PCA)",
        "toolbox.tool_mnf": "Minimum Noise Fraction (MNF)",
        "toolbox.tool_ica": "Independent Component Analysis (ICA)",
        "toolbox.tool_color": "Color Space Transforms (RGB-HSV)",
        "toolbox.tool_sam": "Spectral Angle Mapper (SAM)",
        "toolbox.tool_sid": "Spectral Information Divergence (SID)",
        "toolbox.tool_sff": "Spectral Feature Fitting (SFF)",
        "toolbox.tool_spectral_lib": "Spectral Library Viewer",
        "toolbox.tool_continuum": "Continuum Removal",
        "toolbox.tool_kmeans": "K-Means (Unsupervised)",
        "toolbox.tool_isodata": "ISODATA (Unsupervised)",
        "toolbox.tool_maxlik": "Maximum Likelihood (Supervised)",
        "toolbox.tool_svm": "Support Vector Machine (SVM)",
        "toolbox.tool_accuracy": "Confusion Matrix & Accuracy Assessment",
        "toolbox.tool_ndvi": "Normalized Difference Vegetation Index (NDVI)",
        "toolbox.tool_ndwi": "Normalized Difference Water Index (NDWI)",
        "toolbox.tool_evi": "Enhanced Vegetation Index (EVI)",
        "toolbox.tool_savi": "Soil-Adjusted Vegetation Index (SAVI)",
        "toolbox.tool_nbr": "Normalized Burn Ratio (NBR)",

        # Spectral Profile Dock
        "dock.spectral_profile": "Spectral Profile (Z-Profile)",
        "spectral_profile.no_pixel": "No pixel selected",
        "spectral_profile.cleared": "Spectrum: Cleared",
        "spectral_profile.loaded": "Spectrum Loaded",
        "spectral_profile.btn_clear": "Clear",
        "spectral_profile.axis_x_band": "Band Number",
        "spectral_profile.axis_x_wavelength": "Band / Wavelength",
        "spectral_profile.axis_y": "Value / Reflectance",

        # Main View & Overview
        "main_view.overview": "Overview",

        # Status Bar
        "status.file_coords": "File: --, --",
        "status.geo_coords": "Geo: --, --",
        "status.pixel_value": "Value: --",

        # Dialogs - General
        "dialog.open_title": "Open Remote Sensing Image",
        "dialog.open_filter": "All Supported Images (*.hdr *.dat *.raw *.tif *.tiff *.img);;ENVI Files (*.hdr *.dat *.raw *.img);;GeoTIFF & ERDAS Imagine (*.tif *.tiff *.img);;All Files (*)",
        "dialog.no_active_layer": "Please open or select an active raster layer first.",
        "dialog.no_active_title": "No Active Layer",
        "dialog.btn_ok": "OK",
        "dialog.btn_cancel": "Cancel",
        "dialog.btn_apply": "Apply",

        # Dialogs - Band Math
        "dialog.band_math.title": "Band Math",
        "dialog.band_math.expr": "Enter Mathematical Expression:",
        "dialog.band_math.expr_tip": "Example: (b5 - b4) / (b5 + b4). Variables b1, b2, etc. correspond to bands.",
        "dialog.band_math.assign": "Variable - Band Assignment",
        "dialog.band_math.btn_compute": "Execute Band Math",

        # Dialogs - Spectral Indices
        "dialog.indices.title": "Spectral Indices",
        "dialog.indices.select": "Select Index:",
        "dialog.indices.band_mapping": "Band Assignments",
        "dialog.indices.nir": "Near-Infrared (NIR) Band:",
        "dialog.indices.red": "Red Band:",
        "dialog.indices.green": "Green Band:",
        "dialog.indices.blue": "Blue Band:",
        "dialog.indices.swir": "Shortwave Infrared (SWIR) Band:",
        "dialog.indices.btn_compute": "Compute Index",

        # Dialogs - PCA & MNF
        "dialog.pca.title": "Principal Component Analysis (PCA)",
        "dialog.pca.num_comp": "Number of Components:",
        "dialog.pca.chk_rgb": "Generate False Color Composite (PC1, PC2, PC3)",
        "dialog.pca.report_tip": "Variance and eigenvalue statistics will appear here after execution...",
        "dialog.pca.btn_run": "Compute PCA",

        # Dialogs - Classification
        "dialog.classification.title": "Image Classification (K-Means / ISODATA / SAM)",
        "dialog.classification.method": "Classification Method:",
        "dialog.classification.classes": "Number of Classes:",
        "dialog.classification.max_iter": "Max Iterations:",
        "dialog.classification.threshold": "Change Threshold (%):",
        "dialog.classification.btn_run": "Execute Classification",

        # Dialogs - ROI Tool
        "dialog.roi.title": "Region of Interest (ROI) Tool",
        "dialog.roi.rois": "Regions of Interest:",
        "dialog.roi.btn_add_poly": "Add Polygon ROI",
        "dialog.roi.btn_add_rect": "Add Rect ROI",
        "dialog.roi.btn_stats": "Calculate Statistics",
        "dialog.roi.btn_plot": "Plot Mean Spectrum",
        "dialog.roi.stats_title": "ROI Multi-band Statistics:",
        "dialog.roi.col_band": "Band",
        "dialog.roi.col_pixels": "Pixels",
        "dialog.roi.col_mean": "Mean",
        "dialog.roi.col_stdev": "StDev",
        "dialog.roi.col_min": "Min",
        "dialog.roi.col_max": "Max",

        # Dialogs - Synthetic Generator
        "dialog.synthetic.title": "Generate Synthetic Benchmark Data",
        "dialog.synthetic.lines": "Lines (Height):",
        "dialog.synthetic.samples": "Samples (Width):",
        "dialog.synthetic.bands": "Number of Bands:",
        "dialog.synthetic.noise": "Gaussian Noise (SNR):",
        "dialog.synthetic.btn_generate": "Generate & Load Dataset",
    },
    "zh": {
        # App & Window
        "app.title": "OpenENVI - 遥感与高光谱分析平台",
        "app.ready": "就绪",
        "app.initialized": "OpenENVI 平台初始化完成",
        "app.about_title": "关于 OpenENVI",
        "app.about_desc": (
            "<h3>OpenENVI 遥感与高光谱分析平台</h3>"
            "<p>一款现代化、轻量级、开源的 ENVI 风格桌面遥感与高光谱影像分析平台。</p>"
            "<p><b>Phase 0: 基础架构脚手架与模块化 UI 界面</b></p>"
        ),

        # Menu Titles
        "menu.file": "文件(&F)",
        "menu.view": "视图(&V)",
        "menu.layer": "图层(&L)",
        "menu.display": "显示(&D)",
        "menu.tools": "工具(&T)",
        "menu.window": "窗口(&W)",
        "menu.language": "语言(&G)",
        "menu.help": "帮助(&H)",

        # Menu & Toolbar Actions
        "action.open": "打开影像(&O)...",
        "action.exit": "退出(&X)",
        "action.probe": "探针拾取",
        "action.pan": "平移浏览",
        "action.zoom_in": "放大",
        "action.zoom_out": "缩小",
        "action.fit": "适应窗口",
        "action.reset_layout": "重置面板布局",
        "action.about": "关于 OpenENVI(&A)",
        "action.layer_props": "图层属性...",
        "action.remove_layer": "移除当前活动图层",

        # Stretch Modes
        "stretch.label": " 动态拉伸: ",
        "stretch.linear2": "线性 2% 拉伸",
        "stretch.linear5": "线性 5% 拉伸",
        "stretch.equalize": "直方图均衡化",
        "stretch.gaussian": "高斯分布拉伸",
        "stretch.none": "无拉伸 (原始值)",

        # Layer Manager Dock
        "dock.layer_manager": "图层管理器",
        "layer_manager.col_name": "图层名称",
        "layer_manager.col_type": "显示类型",
        "layer_manager.btn_remove": "移除图层",
        "layer_manager.btn_up": "上移",
        "layer_manager.btn_down": "下移",
        "layer_manager.msg_removed": "图层已移除",

        # Data Manager Dock
        "dock.data_manager": "数据管理器 (可用波段)",
        "data_manager.rb_gray": "单波段灰度",
        "data_manager.rb_rgb": "RGB 彩色合成",
        "data_manager.rgb_group": "RGB 波段通道分配",
        "data_manager.btn_load_band": "加载单波段",
        "data_manager.btn_load_rgb": "加载 RGB 合成",
        "data_manager.btn_close_file": "关闭影像文件",
        "data_manager.msg_select_band": "请先在上方树列表中选择波段",
        "data_manager.msg_assign_rgb": "请依次为 R、G、B 通道分配对应波段",
        "data_manager.msg_file_closed": "影像文件已关闭",

        # Toolbox Dock
        "dock.toolbox": "ENVI 工具箱",
        "toolbox.search_placeholder": "搜索工具与算法...",
        "toolbox.cat_basic": "基础工具 (Basic Tools)",
        "toolbox.cat_transforms": "正交变换 (Transforms)",
        "toolbox.cat_spectral": "高光谱分析 (Spectral)",
        "toolbox.cat_classification": "遥感分类 (Classification)",
        "toolbox.cat_indices": "波谱指数 (Indices)",

        # Toolbox Tools
        "toolbox.tool_band_math": "波段运算 (Band Math)",
        "toolbox.tool_resize": "数据空间/波段裁剪与重采样",
        "toolbox.tool_stacking": "波段组合/层叠 (Layer Stacking)",
        "toolbox.tool_mosaic": "影像掩膜与镶嵌 (Masking & Mosaicking)",
        "toolbox.tool_stats": "栅格像元统计 (Raster Statistics)",
        "toolbox.tool_roi": "感兴趣区分析工具 (ROI Tool)",
        "toolbox.tool_pca": "主成分分析变换 (PCA)",
        "toolbox.tool_mnf": "最小噪声分离变换 (MNF)",
        "toolbox.tool_ica": "独立成分分析 (ICA)",
        "toolbox.tool_color": "色彩空间变换 (RGB-HSV)",
        "toolbox.tool_sam": "光谱角度制图 (SAM)",
        "toolbox.tool_sid": "光谱信息发散度 (SID)",
        "toolbox.tool_sff": "光谱特征拟合 (SFF)",
        "toolbox.tool_spectral_lib": "波谱库浏览查看器",
        "toolbox.tool_continuum": "连续统去除/包络线消除",
        "toolbox.tool_kmeans": "K-Means 无监督聚类分类",
        "toolbox.tool_isodata": "ISODATA 动态聚类分类",
        "toolbox.tool_maxlik": "最大似然法监督分类",
        "toolbox.tool_svm": "支持向量机分类 (SVM)",
        "toolbox.tool_accuracy": "混淆矩阵与分类精度评价",
        "toolbox.tool_ndvi": "归一化植被指数 (NDVI)",
        "toolbox.tool_ndwi": "归一化水体指数 (NDWI)",
        "toolbox.tool_evi": "增强植被指数 (EVI)",
        "toolbox.tool_savi": "土壤调节植被指数 (SAVI)",
        "toolbox.tool_nbr": "归一化燃烧比 (NBR)",

        # Spectral Profile Dock
        "dock.spectral_profile": "波谱剖面 (Z-Profile)",
        "spectral_profile.no_pixel": "未选择像元",
        "spectral_profile.cleared": "波谱曲线: 已清空",
        "spectral_profile.loaded": "波谱曲线: 已加载",
        "spectral_profile.btn_clear": "清空曲线",
        "spectral_profile.axis_x_band": "波段序号",
        "spectral_profile.axis_x_wavelength": "波段序号 / 波长",
        "spectral_profile.axis_y": "反射率 / 辐射亮度值",

        # Main View & Overview
        "main_view.overview": "鹰眼导航视口",

        # Status Bar
        "status.file_coords": "像元坐标: --, --",
        "status.geo_coords": "地理坐标: --, --",
        "status.pixel_value": "像元灰度值: --",

        # Dialogs - General
        "dialog.open_title": "打开遥感影像文件",
        "dialog.open_filter": "所有支持格式 (*.hdr *.dat *.raw *.tif *.tiff *.img);;ENVI 格式 (*.hdr *.dat *.raw *.img);;GeoTIFF 与 ERDAS (*.tif *.tiff *.img);;所有文件 (*)",
        "dialog.no_active_layer": "请先在图层管理器中加载或选定一个活动图层。",
        "dialog.no_active_title": "无活动图层",
        "dialog.btn_ok": "确定",
        "dialog.btn_cancel": "取消",
        "dialog.btn_apply": "应用",

        # Dialogs - Band Math
        "dialog.band_math.title": "波段运算 (Band Math)",
        "dialog.band_math.expr": "输入数学表达式 (如 (b5 - b4) / (b5 + b4)):",
        "dialog.band_math.expr_tip": "支持变量 b1, b2... 及 np.sin, np.log, np.sqrt 等函数。",
        "dialog.band_math.assign": "表达式变量与影像波段匹配映射",
        "dialog.band_math.btn_compute": "执行波段运算并生成图层",

        # Dialogs - Spectral Indices
        "dialog.indices.title": "光谱遥感指数计算 (Spectral Indices)",
        "dialog.indices.select": "选择指数类型:",
        "dialog.indices.band_mapping": "光谱波段匹配",
        "dialog.indices.nir": "近红外 (NIR) 波段:",
        "dialog.indices.red": "红光 (Red) 波段:",
        "dialog.indices.green": "绿光 (Green) 波段:",
        "dialog.indices.blue": "蓝光 (Blue) 波段:",
        "dialog.indices.swir": "短波红外 (SWIR) 波段:",
        "dialog.indices.btn_compute": "计算指数并生成图层",

        # Dialogs - PCA & MNF
        "dialog.pca.title": "主成分分析变换 (PCA)",
        "dialog.pca.num_comp": "输出主成分分量数 (Components):",
        "dialog.pca.chk_rgb": "自动生成假彩色合成图层 (PC1, PC2, PC3)",
        "dialog.pca.report_tip": "协方差、特征值与累计方差贡献率报表将在计算后在此显示...",
        "dialog.pca.btn_run": "开始计算 PCA 变换",

        # Dialogs - Classification
        "dialog.classification.title": "遥感影像分类 (K-Means / ISODATA / SAM)",
        "dialog.classification.method": "分类算法选择:",
        "dialog.classification.classes": "聚类类别数量 (Classes):",
        "dialog.classification.max_iter": "最大迭代次数:",
        "dialog.classification.threshold": "变化收敛阈值 (%):",
        "dialog.classification.btn_run": "执行分类并生成专题图谱",

        # Dialogs - ROI Tool
        "dialog.roi.title": "感兴趣区分析工具 (ROI Tool)",
        "dialog.roi.rois": "感兴趣区列表:",
        "dialog.roi.btn_add_poly": "添加多边形 ROI",
        "dialog.roi.btn_add_rect": "添加矩形 ROI",
        "dialog.roi.btn_stats": "计算多波段像元统计",
        "dialog.roi.btn_plot": "投影 ROI 均值波谱曲线",
        "dialog.roi.stats_title": "ROI 多波段像元统计报表:",
        "dialog.roi.col_band": "波段",
        "dialog.roi.col_pixels": "像元总数",
        "dialog.roi.col_mean": "均值",
        "dialog.roi.col_stdev": "标准差",
        "dialog.roi.col_min": "最小值",
        "dialog.roi.col_max": "最大值",

        # Dialogs - Synthetic Generator
        "dialog.synthetic.title": "生成合成高光谱基准测试数据",
        "dialog.synthetic.lines": "行数 (高度 Lines):",
        "dialog.synthetic.samples": "列数 (宽度 Samples):",
        "dialog.synthetic.bands": "波段数量 (Bands):",
        "dialog.synthetic.noise": "高斯噪声 (信噪比 SNR):",
        "dialog.synthetic.btn_generate": "一键生成并加载数据立方体",
    },
}


class TranslationManager(QObject):
    """Centralized Internationalization Manager."""

    language_changed = Signal(str)

    def __init__(self, default_lang: str = "en"):
        super().__init__()
        self._current_lang = default_lang

    @property
    def current_language(self) -> str:
        """Get currently active language code ('en' or 'zh')."""
        return self._current_lang

    def set_language(self, lang: str) -> None:
        """Switch current language and emit update signal."""
        if lang not in TRANSLATIONS:
            lang = "en"
        if self._current_lang != lang:
            self._current_lang = lang
            self.language_changed.emit(lang)

    def tr(self, key: str, default: Optional[str] = None) -> str:
        """Retrieve translated string for the given key."""
        lang_dict = TRANSLATIONS.get(self._current_lang, TRANSLATIONS["en"])
        if key in lang_dict:
            return lang_dict[key]
        fallback_dict = TRANSLATIONS["en"]
        if key in fallback_dict:
            return fallback_dict[key]
        return default if default is not None else key


# Global singleton instance
i18n = TranslationManager(default_lang="en")


def tr(key: str, default: Optional[str] = None) -> str:
    """Convenience shortcut for i18n.tr()."""
    return i18n.tr(key, default)
