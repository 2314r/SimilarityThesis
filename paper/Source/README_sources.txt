FIGURE & TABLE SOURCE FILES
===========================

This folder holds the source material behind the figures and tables in the manuscript,
as required by the journal-project guidelines.

1. figure_raw_data.xlsx
   Raw data tables (one sheet per table) that back every chart in the paper:
   - retrieval_comparison_clean : main 7-method retrieval results (Table: Main Results)
   - cv_weight_selection        : per-fold cross-validated weight selection (Table: CV)
   - cv_summary / cv_significance: held-out CV summary and significance
   - statistical_significance   : Wilcoxon + Cohen's d (Table: Significance)
   - bootstrap_ci               : bootstrap 95% CIs for all metrics
   - per_family_breakdown       : per-family P@5 and MRR
   - robustness_analysis        : MRR vs semantic-noise level
   - ablation_semantic_features : semantic ablation
   - weight_grid_search         : full weight-grid MRR surface
   - hybrid_weights_final       : final fusion weights
   - feature_correlation        : technical/structural feature correlations

2. Figure-generating source
   All bar charts and graphs are generated programmatically (300 dpi PDF) by the
   notebook ThesisV10.ipynb (see ../Codes/). The notebook is the editable source for
   every data figure; re-running it regenerates the PDFs in ontology/outputs/figures/
   which are mirrored into ../Figures/.

   The schematic workflow figure (workflow.pdf) is the only hand-authored diagram;
   its editable source should be saved here as workflow.pptx (PowerPoint) before
   submission.

Note: data figures are exported directly to PDF at >=300 ppi as required; PNG copies
exist only as previews and are not used in the manuscript.
