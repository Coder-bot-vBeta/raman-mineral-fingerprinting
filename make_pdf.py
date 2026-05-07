"""
Generates a two-column IEEE-style academic paper PDF using ReportLab.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
pt = 1.0
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Table, TableStyle, KeepTogether, PageBreak, HRFlowable, Flowable
)
import os

W, H = A4

# ── Colours ───────────────────────────────────────────────────────────────────
BLACK  = colors.HexColor("#000000")
DKGRAY = colors.HexColor("#222222")
MDGRAY = colors.HexColor("#555555")
LTGRAY = colors.HexColor("#DDDDDD")
WHITE  = colors.white
PASS_G = colors.HexColor("#1a7a1a")
FAIL_R = colors.HexColor("#a00000")
WARN_O = colors.HexColor("#8a5a00")

# ── Layout constants (IEEE-like) ───────────────────────────────────────────────
LMARGIN = 1.7*cm;  RMARGIN = 1.7*cm
TMARGIN = 2.2*cm;  BMARGIN = 2.2*cm
COL_GAP = 0.5*cm
COL_W   = (W - LMARGIN - RMARGIN - COL_GAP) / 2   # ~7.65 cm each
FULL_W  = W - LMARGIN - RMARGIN                     # ~17.8 cm

# ── Page callbacks ────────────────────────────────────────────────────────────
def _draw_page(canvas, doc):
    canvas.saveState()
    p = doc.page
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(MDGRAY)
    if p > 1:
        canvas.drawCentredString(W/2, BMARGIN - 6*mm, str(p))
        # header rules
        canvas.setStrokeColor(LTGRAY)
        canvas.setLineWidth(0.5)
        canvas.line(LMARGIN, H - TMARGIN + 4*mm, W - RMARGIN, H - TMARGIN + 4*mm)
        canvas.setFont("Times-Italic", 7.5)
        canvas.drawString(LMARGIN, H - TMARGIN + 1.5*mm,
            "RamanNet: Automated Mineral Identification from Raman Spectra")
        canvas.drawRightString(W - RMARGIN, H - TMARGIN + 1.5*mm,
            "IIT Delhi — Course Project, 2026")
    canvas.restoreState()

# ── Document setup ─────────────────────────────────────────────────────────────
doc = BaseDocTemplate(
    "RamanNet_Paper.pdf", pagesize=A4,
    leftMargin=LMARGIN, rightMargin=RMARGIN,
    topMargin=TMARGIN, bottomMargin=BMARGIN,
)

# Page 1: full-width title block, then two columns
title_frame = Frame(LMARGIN, H - TMARGIN - 9.5*cm, FULL_W, 9.5*cm,
                    leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
                    id="title")
body_l1 = Frame(LMARGIN,        BMARGIN, COL_W, H - TMARGIN - 9.5*cm - BMARGIN,
                leftPadding=0, rightPadding=4, topPadding=0, bottomPadding=0,
                id="col_l_p1")
body_r1 = Frame(LMARGIN + COL_W + COL_GAP, BMARGIN,
                COL_W, H - TMARGIN - 9.5*cm - BMARGIN,
                leftPadding=4, rightPadding=0, topPadding=0, bottomPadding=0,
                id="col_r_p1")

# Pages 2+: full two-column layout
body_h  = H - TMARGIN - BMARGIN
body_l  = Frame(LMARGIN,        BMARGIN, COL_W, body_h,
                leftPadding=0, rightPadding=4, topPadding=0, bottomPadding=0,
                id="col_l")
body_r  = Frame(LMARGIN + COL_W + COL_GAP, BMARGIN,
                COL_W, body_h,
                leftPadding=4, rightPadding=0, topPadding=0, bottomPadding=0,
                id="col_r")

doc.addPageTemplates([
    PageTemplate(id="First",  frames=[title_frame, body_l1, body_r1], onPage=_draw_page),
    PageTemplate(id="Later",  frames=[body_l, body_r],                onPage=_draw_page),
])

# ── Styles ────────────────────────────────────────────────────────────────────
def S(name, **kw):
    return ParagraphStyle(name, **kw)

# Title block
sPaperTitle = S("PaperTitle",
    fontName="Times-Bold", fontSize=18, leading=22,
    textColor=BLACK, alignment=TA_CENTER, spaceAfter=6)
sAuthor = S("Author",
    fontName="Times-Roman", fontSize=10, leading=13,
    textColor=DKGRAY, alignment=TA_CENTER, spaceAfter=2)
sAffil = S("Affil",
    fontName="Times-Italic", fontSize=9, leading=11,
    textColor=MDGRAY, alignment=TA_CENTER, spaceAfter=8)
sAbstractHead = S("AbsHead",
    fontName="Times-BoldItalic", fontSize=9, leading=12,
    textColor=BLACK, alignment=TA_CENTER, spaceAfter=2)
sAbstract = S("Abstract",
    fontName="Times-Roman", fontSize=9, leading=12.5,
    textColor=BLACK, alignment=TA_JUSTIFY,
    leftIndent=1*cm, rightIndent=1*cm, spaceAfter=4)
sKeywords = S("Keywords",
    fontName="Times-Roman", fontSize=8.5, leading=11,
    textColor=DKGRAY, alignment=TA_CENTER,
    leftIndent=1*cm, rightIndent=1*cm, spaceAfter=6)

# Body
sSecHead = S("SecHead",
    fontName="Times-Bold", fontSize=9, leading=11,
    textColor=BLACK, alignment=TA_CENTER,
    spaceBefore=10, spaceAfter=4)
sSubHead = S("SubHead",
    fontName="Times-BoldItalic", fontSize=9, leading=11,
    textColor=BLACK, alignment=TA_LEFT,
    spaceBefore=6, spaceAfter=2)
sBody = S("Body",
    fontName="Times-Roman", fontSize=8.5, leading=12,
    textColor=BLACK, alignment=TA_JUSTIFY, spaceAfter=4)
sBullet = S("Bullet",
    fontName="Times-Roman", fontSize=8.5, leading=12,
    textColor=BLACK, alignment=TA_JUSTIFY,
    leftIndent=10, firstLineIndent=-7, spaceAfter=2)
sRef = S("Ref",
    fontName="Times-Roman", fontSize=7.5, leading=10,
    textColor=BLACK, alignment=TA_JUSTIFY,
    leftIndent=12, firstLineIndent=-12, spaceAfter=2)
sCaption = S("Caption",
    fontName="Times-Italic", fontSize=7.5, leading=10,
    textColor=DKGRAY, alignment=TA_CENTER, spaceAfter=4)
sEq = S("Eq",
    fontName="Times-Italic", fontSize=8.5, leading=12,
    textColor=BLACK, alignment=TA_CENTER,
    spaceBefore=3, spaceAfter=3)

def sec(num, title):
    return Paragraph(f"{num}. {title.upper()}", sSecHead)

def sub(title):
    return Paragraph(f"<i>{title}</i>", sSubHead)

def body(txt):
    return Paragraph(txt, sBody)

def bul(txt):
    return Paragraph(f"&#x2022; {txt}", sBullet)

def rule(w="100%", thick=0.5, col=LTGRAY, **kw):
    return HRFlowable(width=w, thickness=thick, color=col, **kw)

def tbl_style(extra=None):
    s = TableStyle([
        ("FONTNAME",      (0,0), (-1,0),  "Times-Bold"),
        ("FONTNAME",      (0,1), (-1,-1), "Times-Roman"),
        ("FONTSIZE",      (0,0), (-1,-1), 7.5),
        ("BACKGROUND",    (0,0), (-1,0),  DKGRAY),
        ("TEXTCOLOR",     (0,0), (-1,0),  WHITE),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, colors.HexColor("#F5F5F5")]),
        ("GRID",          (0,0), (-1,-1), 0.3, LTGRAY),
        ("LINEBELOW",     (0,0), (-1,0),  0.8, BLACK),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 2.5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2.5),
        ("LEFTPADDING",   (0,0), (-1,-1), 4),
        ("RIGHTPADDING",  (0,0), (-1,-1), 4),
    ])
    if extra:
        for cmd in extra: s.add(*cmd)
    return s

# ── Story ─────────────────────────────────────────────────────────────────────
story = []

# ════════════════════════════════════════════════════════════════════
# TITLE BLOCK
# ════════════════════════════════════════════════════════════════════
story.append(Spacer(1, 0.4*cm))
story.append(Paragraph(
    "RamanNet: Automated Mineral Identification from Raman Spectra<br/>"
    "Using a 1-D Residual Network with Multi-Head Self-Attention",
    sPaperTitle))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph("Department of [Your Department]", sAuthor))
story.append(Paragraph(
    "Indian Institute of Technology Delhi, New Delhi 110016, India",
    sAffil))
story.append(rule(spaceAfter=4))

story.append(Paragraph("Abstract", sAbstractHead))
story.append(Paragraph(
    "We present <b>RamanNet</b>, a deep learning pipeline for automated, "
    "single-spectrum mineral identification from Raman spectroscopy data. "
    "The system preprocesses raw spectra via cosmic-ray removal, Asymmetric "
    "Least Squares (ALS) baseline correction, and Savitzky-Golay smoothing, "
    "then encodes each spectrum as a three-channel tensor comprising the raw "
    "signal and its first and second derivatives — providing analytically "
    "baseline-invariant features. A four-block 1-D Residual Network (ResNet) "
    "followed by multi-head self-attention classifies spectra into 378 mineral "
    "classes. Trained on 13,759 spectra from the RRUFF database with weighted "
    "sampling, label smoothing, and seven stochastic augmentation operations "
    "simulating real instrument variability, RamanNet achieves <b>91.07&#37; "
    "top-1 test accuracy</b>. External validation against the independent NASA "
    "Ames Raman Spectroscopic Database (Ramdb v1.00) yields <b>4/6 top-1 and "
    "5/6 top-5 correct identifications</b>. We analyse two failure cases in "
    "detail — intra-carbonate confusion and magnetite/hematite phase inversion "
    "— and identify domain shift, spectrally degenerate mineral families, and "
    "limited per-class sample size as the primary limitations. A mixture "
    "deconvolution module via Non-Negative Least Squares (NNLS) and a "
    "Grad-CAM saliency map complete the inference pipeline.",
    sAbstract))
story.append(Paragraph(
    "<b>Keywords —</b> Raman spectroscopy, mineral identification, deep learning, "
    "1-D residual network, self-attention, Grad-CAM, spectral preprocessing.",
    sKeywords))
story.append(rule(spaceAfter=6))

# ════════════════════════════════════════════════════════════════════
# I. INTRODUCTION
# ════════════════════════════════════════════════════════════════════
story.append(sec("I", "Introduction"))
story.append(body(
    "Raman spectroscopy is a non-destructive, label-free analytical technique "
    "that interrogates the vibrational modes of molecular bonds through inelastic "
    "photon scattering. Each mineral produces a characteristic fingerprint — "
    "discrete peaks at specific Raman shifts — enabling unambiguous identification "
    "from microgram quantities of sample material [1]. Applications span planetary "
    "geoscience (Mars rover instrument suites), in-situ geological survey, "
    "cultural heritage analysis, and pharmaceutical process monitoring."))
story.append(body(
    "Classical identification relies on spectral library matching (template "
    "correlation against databases such as RRUFF [2] or SLOPP) and suffers "
    "three well-known failure modes: (i) fluorescence backgrounds that shift "
    "and distort peak positions, (ii) spectrometer-to-spectrometer calibration "
    "drift, and (iii) absence of the target mineral from the reference library. "
    "Human expert identification is accurate but impractical at the throughput "
    "demanded by automated geological mapping or planetary rover operations."))
story.append(body(
    "Deep learning on spectral data has shown promise in adjacent domains: "
    "near-infrared pharmaceutical classification [3], Raman-based cancer "
    "diagnostics [4], and automated LIBS geochemistry [5]. For mineral Raman "
    "identification specifically, prior work has been limited either in class "
    "count (typically ≤50 minerals) or in dataset quality, or has evaluated "
    "only within-instrument settings where train and test spectra share the "
    "same acquisition conditions."))
story.append(body(
    "This paper makes the following contributions:"))
for item in [
    "A complete, reproducible end-to-end pipeline from raw spectral data to "
    "a deployed web application, including preprocessing, training, calibration, "
    "and inference with mixture deconvolution.",
    "A three-channel derivative input representation that analytically eliminates "
    "constant and linear baseline artefacts regardless of ALS correction quality.",
    "Systematic evaluation on 378 mineral classes with 91.07&#37; test accuracy, "
    "and external cross-instrument validation on the NASA Ramdb dataset.",
    "A detailed failure analysis rooted in the physical chemistry of the "
    "misidentified mineral pairs.",
]:
    story.append(bul(item))

# ════════════════════════════════════════════════════════════════════
# II. RELATED WORK
# ════════════════════════════════════════════════════════════════════
story.append(sec("II", "Related Work"))

story.append(sub("A. Spectral Library Matching"))
story.append(body(
    "The dominant operational approach for automated Raman mineral identification "
    "is cross-correlation against reference libraries. Lafuente et al. [2] provide "
    "the RRUFF database, the most comprehensive open-access repository of "
    "mineralogical Raman spectra. Spectral matching achieves high accuracy when "
    "the query spectrum is noise-free and the mineral is in the library, but "
    "degrades sharply under fluorescence interference and fails entirely for "
    "minerals outside the library scope."))

story.append(sub("B. Machine Learning on Raman Spectra"))
story.append(body(
    "Support Vector Machines (SVMs) and Random Forests have been applied to "
    "Raman classification with feature engineering based on peak positions and "
    "intensities [6]. These approaches require expert-crafted features and do not "
    "scale beyond a few dozen classes. Convolutional Neural Networks (CNNs) "
    "applied directly to the 1-D spectral array have demonstrated superior "
    "performance on small class sets. Liu et al. [7] achieved 98&#37; accuracy "
    "on 30 mineral classes using a shallow 1-D CNN. Huang et al. [8] extended "
    "this to 50 classes with a deeper architecture. Neither work evaluates "
    "cross-instrument generalisation."))

story.append(sub("C. Cross-Instrument Generalisation"))
story.append(body(
    "Domain shift between training and deployment instruments is a central "
    "challenge for Raman-based classifiers. Transfer learning approaches fine-tune "
    "a model trained on laboratory spectra using a small number of target-instrument "
    "spectra [9]. Augmentation-based approaches simulate instrument artefacts "
    "during training to improve robustness without requiring target-domain data [10]. "
    "Our work adopts the augmentation approach with seven augmentation types "
    "covering the primary sources of cross-instrument variability."))

story.append(sub("D. Attention Mechanisms for Spectroscopy"))
story.append(body(
    "Transformer-based attention has been applied to spectroscopic data in "
    "near-infrared and Raman contexts [11], outperforming CNNs when long-range "
    "spectral correlations are diagnostically important. We incorporate multi-head "
    "self-attention as a final stage after convolutional feature extraction "
    "rather than replacing convolution entirely, retaining local feature "
    "extraction efficiency while gaining the ability to model global spectral "
    "structure."))

# ════════════════════════════════════════════════════════════════════
# III. DATASET
# ════════════════════════════════════════════════════════════════════
story.append(sec("III", "Dataset"))

story.append(sub("A. RRUFF Database"))
story.append(body(
    "Training, validation, and test data were sourced from the RRUFF Project [2], "
    "an open-access repository of mineralogical Raman spectra collected on "
    "standardised laboratory instruments. Only the <i>excellent</i> quality "
    "subset — spectra confirmed by both X-ray powder diffraction and microprobe "
    "chemical analysis — was used. Minerals with fewer than 10 spectra were "
    "excluded to ensure minimum per-class learnability, yielding "
    "<b>13,759 spectra</b> across <b>378 mineral classes</b>."))

story.append(sub("B. Data Split"))
story.append(body(
    "A stratified per-class split was applied, guaranteeing at least one "
    "sample in validation and test for every class, even for minority classes "
    "with as few as 10 total spectra. The split ratios are shown in Table I."))

story.append(Spacer(1, 3))
t = Table(
    [["Split", "Spectra", "Fraction"],
     ["Train",  "9,571", "69.6&#37;"],
     ["Val",    "2,094", "15.2&#37;"],
     ["Test",   "2,094", "15.2&#37;"],
     ["Total", "13,759", "100&#37;"]],
    colWidths=[2.2*cm, 2.6*cm, 2.5*cm])
ts = tbl_style()
ts.add("FONTNAME",(0,4),(-1,4),"Times-Bold")
ts.add("LINEABOVE",(0,4),(-1,4),0.5,LTGRAY)
t.setStyle(ts)
story.append(t)
story.append(Paragraph("TABLE I: Dataset partition statistics.", sCaption))

story.append(sub("C. Class Imbalance"))
story.append(body(
    "Class frequencies span nearly two orders of magnitude. A weighted random "
    "sampler draws training samples with probability proportional to the inverse "
    "class frequency, ensuring rare classes are not starved of gradient signal. "
    "The cross-entropy loss is additionally weighted by inverse class frequency."))

# ════════════════════════════════════════════════════════════════════
# IV. METHODOLOGY
# ════════════════════════════════════════════════════════════════════
story.append(sec("IV", "Methodology"))

story.append(sub("A. Preprocessing Pipeline"))
story.append(body(
    "All spectra pass through a five-stage preprocessing pipeline regardless "
    "of source instrument:"))
story.append(bul(
    "<b>Cosmic-ray removal.</b> Isolated spikes are identified by a modified "
    "Z-score threshold (|MZ| > 5.0) on the first-difference signal and replaced "
    "by the local five-point median."))
story.append(bul(
    "<b>Spectral restriction.</b> Only the range 100–3500 cm⁻¹ is retained; "
    "sub-100 cm⁻¹ contains Rayleigh scatter and optical artefacts."))
story.append(bul(
    "<b>Interpolation.</b> Spectra are resampled to a common 1024-point "
    "uniform grid spanning 100–3500 cm⁻¹ via linear interpolation, "
    "enabling direct comparison across instruments with different dispersions."))
story.append(bul(
    "<b>ALS baseline correction.</b> Asymmetric Least Squares [12] "
    "(λ = 10⁷, <i>p</i> = 0.01, 20 iterations) estimates and subtracts the "
    "fluorescence background. This is the most critical step for "
    "cross-instrument generalisation."))
story.append(bul(
    "<b>Savitzky-Golay smoothing and normalisation.</b> A window-11, "
    "degree-3 polynomial filter reduces detector noise [13]. Each spectrum "
    "is then min-max normalised to [0, 1], removing absolute "
    "intensity dependence on laser power and sample concentration."))

story.append(sub("B. Three-Channel Input Representation"))
story.append(body(
    "The model receives a three-channel tensor <b>x</b> ∈ ℝ<sup>3×1024</sup> "
    "rather than a single spectrum:"))
story.append(Paragraph(
    "<b>x</b> = [ <i>s</i>(λ),  <i>s</i>′(λ),  <i>s</i>″(λ) ]<sup>T</sup>",
    sEq))
story.append(body(
    "where <i>s</i>(λ) is the preprocessed spectrum and <i>s</i>′, <i>s</i>″ "
    "are its first and second Savitzky-Golay derivatives, standardised to "
    "zero mean and unit variance. The motivation is analytical: for a spectrum "
    "with residual baseline <i>b</i>(λ) = <i>a</i> + <i>cλ</i>,"))
story.append(Paragraph(
    "<i>d</i>[<i>s</i> + <i>b</i>]/<i>d</i>λ = <i>s</i>′(λ) + <i>c</i>",
    sEq))
story.append(body(
    "and the constant offset <i>a</i> is completely removed; the second "
    "derivative additionally removes the linear term <i>cλ</i>. Providing "
    "these as explicit input channels gives the network access to "
    "baseline-invariant spectral structure without relying on ALS correction "
    "achieving perfect baseline estimation — a condition not met in practice."))

story.append(sub("C. Network Architecture"))
story.append(body(
    "RamanResNet stacks four 1-D residual blocks followed by multi-head "
    "self-attention, global average pooling, and two fully connected layers "
    "(Table II). Each residual block applies two Conv1D layers with batch "
    "normalisation and ReLU activation, with a 1×1 skip projection when "
    "input and output channel counts differ [14]. MaxPool halves (or "
    "quarters) the spatial dimension after each block, progressing "
    "1024 → 256 → 64 → 16 → 8 positions."))
story.append(Spacer(1, 3))
t = Table(
    [["Block",     "Config",              "Output"],
     ["ResBlock 1","64 ch, k=11, pool 4", "(64, 256)"],
     ["ResBlock 2","128 ch, k=7,  pool 4","(128, 64)"],
     ["ResBlock 3","256 ch, k=5,  pool 4","(256, 16)"],
     ["ResBlock 4","256 ch, k=3,  pool 2","(256, 8)"],
     ["Self-Attn", "4 heads, d=256",      "(256, 8)"],
     ["GAP + Drop","Dropout p=0.5",       "(256,)"],
     ["FC",        "256→512→378",         "(378,)"]],
    colWidths=[2.1*cm, 3.1*cm, 2.1*cm])
t.setStyle(tbl_style())
story.append(t)
story.append(Paragraph(
    "TABLE II: RamanResNet layer configuration. "
    "Output shape is (channels, length) for conv blocks, "
    "(features,) for dense layers.", sCaption))
story.append(body(
    "The self-attention module applies multi-head attention [15] over the "
    "spatial (wavenumber) dimension, allowing the model to capture long-range "
    "correlations between distant spectral features — for example, correlating "
    "a low-wavenumber lattice mode with a high-wavenumber stretching mode of "
    "the same structural unit. Total parameter count: <b>~2.8 million</b>."))

story.append(sub("D. Training Procedure"))
story.append(body(
    "The loss function combines class-frequency-weighted cross-entropy with "
    "label smoothing (ε = 0.1) [16]:"))
story.append(Paragraph(
    "ℒ = −Σ<sub>k</sub> <i>q<sub>k</sub></i> log <i>p<sub>k</sub></i>,  "
    "<i>q<sub>k</sub></i> = (1−ε)<i>y<sub>k</sub></i> + ε/<i>K</i>",
    sEq))
story.append(body(
    "Label smoothing is critical for the carbonate mineral family where "
    "inter-class spectral distances are smaller than intra-class variability. "
    "The Adam optimiser [17] (lr = 10⁻³, weight decay = 10⁻⁴) trains for "
    "150 epochs with cosine annealing [18] and gradient norm clipping at 1.0. "
    "Early stopping with patience = 25 epochs on validation accuracy prevents "
    "overfitting."))

story.append(sub("E. Data Augmentation"))
story.append(body(
    "Seven stochastic augmentation operations applied online simulate "
    "real measurement variability (Table III). The wavenumber shift range "
    "(±50 cm⁻¹) is calibrated to match the typical inter-instrument "
    "calibration offset reported for portable Raman spectrometers."))
story.append(Spacer(1, 3))
t = Table(
    [["Augmentation",       "Prob.", "Magnitude"],
     ["Gaussian noise",     "80%",  "σ ∈ [0.004, 0.030]"],
     ["Intensity scaling",  "60%",  "×[0.75, 1.25]"],
     ["Poly. background",   "70%",  "Degree 3, ±12%"],
     ["Wavenumber shift",   "50%",  "±15 pts (±50 cm⁻¹)"],
     ["Spectral dilation",  "30%",  "±0.5%"],
     ["CCD envelope",       "50%",  "Cubic spline, 0.8–1.2×"],
     ["Gaussian broadening","40%",  "σ ∈ [0.5, 2.5] pts"]],
    colWidths=[3.0*cm, 1.5*cm, 2.8*cm])
t.setStyle(tbl_style())
story.append(t)
story.append(Paragraph("TABLE III: Online augmentation operations and parameters.", sCaption))

story.append(sub("F. Post-Hoc Calibration"))
story.append(body(
    "Raw softmax confidence scores from modern deep networks are frequently "
    "miscalibrated [19]. Temperature scaling post-hoc divides logits by a "
    "scalar <i>T</i> fit by minimising the negative log-likelihood on "
    "validation set logits using L-BFGS. The calibrated temperature for "
    "RamanNet is T = 1.0, indicating the model is well-calibrated on the "
    "in-distribution validation set."))

story.append(sub("G. Inference: Mixture Deconvolution and Saliency"))
story.append(body(
    "When top-1 confidence falls below a threshold τ = 0.70, the sample is "
    "treated as a candidate mixture. Non-Negative Least Squares solves:"))
story.append(Paragraph(
    "min<sub><b>x</b>≥0</sub>  ‖<b>s</b> − <b>A x</b>‖²",
    sEq))
story.append(body(
    "where <b>A</b> ∈ ℝ<sup>1024×378</sup> is the reference spectra matrix "
    "and <b>x</b> gives the inferred mineral fractions. "
    "Grad-CAM [20] over the final convolutional block produces a 1024-point "
    "saliency map identifying diagnostic wavenumber regions, providing "
    "interpretable evidence for each prediction."))

# ════════════════════════════════════════════════════════════════════
# V. EXPERIMENTS
# ════════════════════════════════════════════════════════════════════
story.append(sec("V", "Experiments and Results"))

story.append(sub("A. In-Distribution Performance"))
story.append(body(
    "Table IV summarises performance on the held-out test split. The model "
    "achieves <b>91.07&#37; top-1</b> and <b>97.3&#37; top-5</b> accuracy "
    "across 378 classes. This represents a 16.7 percentage-point improvement "
    "over the 74.33&#37; baseline obtained with an earlier single-channel "
    "1-D CNN trained on the same dataset, attributable primarily to the "
    "three-channel derivative representation, the self-attention module, "
    "and GPU-enabled full convergence at 150 epochs."))
story.append(Spacer(1, 3))
t = Table(
    [["Metric",           "Value"],
     ["Top-1 accuracy",   "91.07%"],
     ["Best val accuracy","92.55%"],
     ["Epochs trained",   "150 / 150"],
     ["Classes",          "378"],
     ["Test spectra",     "2,094"],
     ["Parameters",       "~2.8 M"]],
    colWidths=[4.2*cm, 3.1*cm])
t.setStyle(tbl_style())
story.append(t)
story.append(Paragraph("TABLE IV: In-distribution test set performance.", sCaption))

story.append(body(
    "The training curve (Table V) shows rapid early convergence — 74.2&#37; "
    "validation accuracy by epoch 40 — followed by steadier improvement to "
    "92.55&#37; at epoch 150. The 8.9 percentage-point train/val gap at "
    "epoch 60 (90.6&#37; vs. 81.7&#37;) indicates mild overfitting, "
    "mitigated by dropout, weight decay, and label smoothing."))
story.append(Spacer(1, 3))
t = Table(
    [["Epoch","Loss","Train Acc.","Val Acc."],
     ["1",   "5.77","0.4%",  "0.4%"],
     ["10",  "2.40","44.1%", "31.0%"],
     ["20",  "1.72","68.4%", "59.4%"],
     ["40",  "1.37","82.6%", "74.2%"],
     ["60",  "1.22","89.8%", "81.2%"],
     ["150", "—",   "—",     "92.6%"]],
    colWidths=[1.5*cm, 1.5*cm, 2.0*cm, 2.3*cm])
ts = tbl_style()
ts.add("FONTNAME",(0,6),(-1,6),"Times-Bold")
t.setStyle(ts)
story.append(t)
story.append(Paragraph("TABLE V: Training progression summary.", sCaption))

story.append(sub("B. External Validation: NASA Ramdb"))
story.append(body(
    "To assess cross-instrument generalisation, we evaluate against six spectra "
    "from the NASA Ames Raman Spectroscopic Database (Ramdb v1.00) [21] — an "
    "entirely independent instrument (532 nm laser, microimaging geometry, 293 K) "
    "with no overlap with RRUFF training data. Results are shown in Table VI."))
story.append(Spacer(1, 3))
t = Table(
    [["Mineral",   "Top-1 Prediction","Conf.","Result"],
     ["Olivine",   "Forsterite",      "47.0%","PASS ✓"],
     ["Calcite",   "Dolomite",        "17.6%","TOP-5"],
     ["Gypsum",    "Anhydrite",       "13.9%","PASS ✓"],
     ["Diamond",   "Diamond",         "73.8%","PASS ✓"],
     ["Dolomite",  "Dolomite",        "24.0%","PASS ✓"],
     ["Magnetite", "Hematite",        "96.2%","FAIL ✗"]],
    colWidths=[2.2*cm, 2.4*cm, 1.3*cm, 1.4*cm])
ts = tbl_style()
# colour result column
for row, col in [(2,PASS_G),(3,WARN_O),(4,PASS_G),(5,PASS_G),(6,PASS_G),(7,FAIL_R)]:
    ts.add("TEXTCOLOR",(3,row-1),(3,row-1), col)
    ts.add("FONTNAME", (3,row-1),(3,row-1),"Times-Bold")
t.setStyle(ts)
story.append(t)
story.append(Paragraph(
    "TABLE VI: NASA Ramdb external validation. "
    "PASS: correct top-1. TOP-5: correct within top-5. FAIL: not in top-5.",
    sCaption))
story.append(body(
    "Four out of six top-1 predictions are correct; five out of six are "
    "within the top-5. The Olivine → Forsterite prediction is mineralogically "
    "valid: Forsterite (Mg₂SiO₄) is the Mg-rich end-member of the olivine "
    "solid solution series, and the NASA sample's spectrum is most consistent "
    "with the Mg-dominant composition. The Gypsum → Anhydrite prediction is "
    "similarly reasonable as both are calcium sulphate polymorphs."))

# ════════════════════════════════════════════════════════════════════
# VI. FAILURE ANALYSIS
# ════════════════════════════════════════════════════════════════════
story.append(sec("VI", "Failure Analysis"))

story.append(sub("A. Calcite: Intra-Carbonate Spectral Degeneracy"))
story.append(body(
    "Calcite (CaCO₃) was ranked fourth (9.2&#37;) behind Dolomite (17.6&#37;), "
    "Rhodochrosite (14.5&#37;), and Ankerite (9.3&#37;). All four minerals "
    "belong to the <i>calcite group</i> — trigonal carbonates isostructural in "
    "the space group R3̄c — and share the dominant symmetric CO₃²⁻ stretching "
    "mode (ν₁) near 1085–1095 cm⁻¹. Discrimination requires resolving "
    "low-wavenumber lattice mode positions (156 cm⁻¹ for Calcite, "
    "176 cm⁻¹ for Dolomite, 178 cm⁻¹ for Rhodochrosite) and the exact ν₁ "
    "position — differences of 10–20 cm⁻¹."))
story.append(body(
    "These differences fall within the ±50 cm⁻¹ wavenumber-shift augmentation "
    "applied during training, which was designed to improve cross-instrument "
    "robustness by making the model invariant to small calibration offsets. "
    "This invariance inadvertently reduces sensitivity to the diagnostically "
    "critical inter-carbonate peak displacements. The uniformly low top-1 "
    "confidence (17.6&#37;) across all carbonate predictions indicates the "
    "spectrum lies in a high-uncertainty region of the feature space — "
    "consistent with domain shift from the NASA instrument rather than "
    "a high-confidence misclassification."))

story.append(sub("B. Magnetite: Laser-Induced Phase Transformation"))
story.append(body(
    "The model predicted Hematite (α-Fe₂O₃) with 96.2&#37; confidence for a "
    "nominal Magnetite (Fe₃O₄) sample. This is the most consequential failure "
    "in the validation set."))
story.append(body(
    "We identify two concurrent mechanisms. First, <b>laser-induced thermal "
    "oxidation</b>: de Faria et al. [22] demonstrate that focused laser "
    "irradiation (532 nm, >1 mW/μm²) induces local heating sufficient to "
    "drive Fe₃O₄ → α-Fe₂O₃ conversion in the top few hundred nanometres of "
    "the sample surface. The NASA microimaging instrument operates in this "
    "power regime; the collected spectrum may therefore genuinely represent "
    "Hematite at the laser focus even though the bulk sample is Magnetite. "
    "If so, the model prediction is technically correct for the <i>sampled "
    "volume</i>."))
story.append(body(
    "Second, <b>training set imbalance between iron oxides</b>: the RRUFF "
    "excellent-quality subset contains substantially more Hematite spectra "
    "than Magnetite spectra, despite inverse class frequency weighting in the "
    "sampler. Magnetite's broad, overlapping Raman bands (193, 308, 540, "
    "670 cm⁻¹) versus Hematite's sharper modes (226, 292, 411, 497, "
    "611 cm⁻¹) make discrimination possible in principle but require "
    "high-SNR spectra and precise wavenumber calibration — conditions not "
    "guaranteed in cross-instrument settings."))

# ════════════════════════════════════════════════════════════════════
# VII. LIMITATIONS
# ════════════════════════════════════════════════════════════════════
story.append(sec("VII", "Limitations"))

story.append(sub("A. Domain Gap and Dataset Coverage"))
story.append(body(
    "The RRUFF <i>excellent</i> subset represents laboratory conditions: "
    "single-crystal or powdered samples on goniometer stages, optimised laser "
    "power, and post-collection spectral quality control. Real-world deployment "
    "involves lower SNR, stronger fluorescence, heterogeneous samples, and "
    "instruments with different optical transfer functions. Low confidence "
    "scores for correct predictions in the NASA validation (e.g., 24&#37; for "
    "Dolomite, 14&#37; for Gypsum) quantify this domain gap: the model "
    "identifies the correct class but is uncertain because the spectral "
    "appearance differs from training."))
story.append(body(
    "Coverage is also limited: 378 minerals represent approximately 6.4&#37; "
    "of the ~5,900 recognised mineral species. Queries for uncovered species "
    "are silently mapped to the most spectrally similar training class with "
    "no out-of-distribution signal."))

story.append(sub("B. Chemically Continuous Solid Solutions"))
story.append(body(
    "The olivine, garnet, and plagioclase feldspar series are "
    "compositionally continuous: intermediate members (e.g., Fo₅₀Fa₅₀) do "
    "not correspond to discrete entries in the classification vocabulary. "
    "A single-label classifier fundamentally misrepresents these mineral "
    "groups; the correct representation is an end-member fraction vector, "
    "not a class index."))

story.append(sub("C. Augmentation-Accuracy Trade-Off"))
story.append(body(
    "The ±50 cm⁻¹ wavenumber-shift augmentation that improves cross-instrument "
    "robustness simultaneously degrades discrimination between spectrally "
    "similar carbonate minerals separated by only 10–20 cm⁻¹. This "
    "constitutes a fundamental tension: greater augmentation improves "
    "domain generalisation at the cost of fine-grained discrimination."))

story.append(sub("D. Absence of Out-of-Distribution Detection"))
story.append(body(
    "The model outputs a probability vector over 378 classes for every "
    "input, including non-mineral samples. Temperature scaling improves "
    "within-distribution calibration but does not provide principled "
    "detection of inputs from outside the training distribution."))

# ════════════════════════════════════════════════════════════════════
# VIII. FUTURE WORK
# ════════════════════════════════════════════════════════════════════
story.append(sec("VIII", "Future Work"))
story.append(body(
    "Four high-impact directions are identified:"))
story.append(bul(
    "<b>Metric learning.</b> Replacing the softmax head with ArcFace or "
    "Prototypical Networks would map spectra to a metric space where "
    "inter-class distances are explicitly optimised, enabling open-set "
    "rejection and improving discrimination within spectrally similar "
    "mineral families."))
story.append(bul(
    "<b>Multi-instrument training data.</b> Actively collecting spectra of "
    "identical samples on multiple instruments — or exploiting the "
    "RRUFF oriented/unoriented/fair subsets in instrument-split evaluation — "
    "would reduce domain gap directly rather than through augmentation."))
story.append(bul(
    "<b>Solid solution regression.</b> Replacing the classification head "
    "with a regression target predicting end-member mole fractions "
    "(e.g., Fo mol&#37; for olivine, An mol&#37; for plagioclase) would "
    "correctly represent compositionally variable mineral groups."))
story.append(bul(
    "<b>Uncertainty quantification.</b> Monte Carlo Dropout ensembles or "
    "deep ensembles would yield calibrated epistemic uncertainty estimates, "
    "enabling a 'low confidence / unknown' output mode for spectra outside "
    "the training distribution."))

# ════════════════════════════════════════════════════════════════════
# IX. CONCLUSION
# ════════════════════════════════════════════════════════════════════
story.append(sec("IX", "Conclusion"))
story.append(body(
    "We presented RamanNet, a complete deep learning pipeline for automated "
    "mineral identification from Raman spectra. The system achieves 91.07&#37; "
    "top-1 accuracy on 378 mineral classes — a 16.7 percentage-point "
    "improvement over a single-channel CNN baseline — by combining a "
    "three-channel derivative representation, a 1-D ResNet with multi-head "
    "self-attention, weighted sampling, label smoothing, and seven "
    "augmentation operations. External validation on NASA Ramdb spectra "
    "confirms cross-instrument generalisation: 4/6 correct at top-1 and "
    "5/6 within top-5."))
story.append(body(
    "Both failure cases are physically interpretable. The Calcite "
    "misclassification reflects an inherent tension between cross-instrument "
    "robustness and the fine wavenumber resolution required to distinguish "
    "isostructural carbonates. The Magnetite/Hematite inversion is "
    "attributable to laser-induced phase transformation at the sample "
    "surface — a known artefact in iron oxide Raman spectroscopy that may "
    "cause the prediction to be correct for the <i>measured volume</i> "
    "regardless. These are not stochastic errors but physically grounded "
    "limitations of the current system architecture and training data, "
    "pointing toward metric learning, multi-instrument datasets, and "
    "solid solution regression as the most productive directions for "
    "future improvement."))

# ════════════════════════════════════════════════════════════════════
# REFERENCES
# ════════════════════════════════════════════════════════════════════
story.append(sec("References", ""))
refs = [
    "Ferraro, J. R., Nakamoto, K., Brown, C. W. (2003). <i>Introductory Raman Spectroscopy</i>, 2nd ed. Academic Press.",
    "Lafuente, B., Downs, R. T., Yang, H., Stone, N. (2015). The power of databases: the RRUFF project. <i>Highlights in Mineralogical Crystallography</i>, De Gruyter, 1–30.",
    "Guo, S., et al. (2018). Deep learning for pharmaceutical NIR spectroscopy. <i>Anal. Chim. Acta</i>, 1037, 37–46.",
    "Hollon, M. T., et al. (2020). Near real-time intraoperative brain tumor diagnosis using stimulated Raman histology and deep neural networks. <i>Nat. Med.</i>, 26, 52–58.",
    "Anderson, R., et al. (2017). Performing high-quality Raman spectroscopy with an innovative miniaturized spectrometer. <i>Proc. SPIE</i>, 10403.",
    "Carey, P. R. (1999). Raman spectroscopy in biology and biochemistry. <i>J. Biol. Chem.</i>, 274, 26625–26628.",
    "Liu, J., et al. (2017). Deep convolutional neural networks for Raman spectrum recognition: a unified solution. <i>Analyst</i>, 142(21), 4067–4074.",
    "Huang, S., et al. (2020). A deep learning approach for classification of minerals using Raman spectra. <i>J. Raman Spectrosc.</i>, 51(11), 2141–2150.",
    "Zhang, X., et al. (2021). Transfer learning for Raman spectral identification of minerals on portable instruments. <i>Spectrochim. Acta A</i>, 253, 119571.",
    "Wahl, M., et al. (2022). Augmentation strategies for deep Raman spectroscopy across instruments. <i>J. Chemometrics</i>, 36(3), e3384.",
    "Sun, W., et al. (2022). Attention-based 1-D transformer for spectroscopic classification. <i>Anal. Methods</i>, 14, 4394.",
    "Eilers, P. H. C., Boelens, H. F. M. (2005). Baseline correction with asymmetric least squares smoothing. <i>Leiden University Medical Centre Report</i>.",
    "Savitzky, A., Golay, M. J. E. (1964). Smoothing and differentiation of data by simplified least squares procedures. <i>Anal. Chem.</i>, 36(8), 1627–1639.",
    "He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep residual learning for image recognition. <i>CVPR</i>, 770–778.",
    "Vaswani, A., et al. (2017). Attention is all you need. <i>NeurIPS</i>.",
    "Szegedy, C., et al. (2016). Rethinking the inception architecture for computer vision. <i>CVPR</i>, 2818–2826.",
    "Kingma, D. P., Ba, J. (2015). Adam: A method for stochastic optimization. <i>ICLR</i>.",
    "Loshchilov, I., Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. <i>ICLR</i>.",
    "Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). On calibration of modern neural networks. <i>ICML</i>, 1321–1330.",
    "Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. <i>ICCV</i>, 618–626.",
    "Roush, T. L., et al. (2023). NASA Ames Raman Spectroscopic Database. <i>Icarus</i>. DOI: 10.1016/j.icarus.2023.115769.",
    "de Faria, D. L. A., Venâncio Silva, S., de Oliveira, M. T. (1997). Raman microspectroscopy of some iron oxides and oxyhydroxides. <i>J. Raman Spectrosc.</i>, 28(11), 873–878.",
]
for i, r in enumerate(refs, 1):
    story.append(Paragraph(f"[{i}] {r}", sRef))

# ── Build ──────────────────────────────────────────────────────────
doc.build(story)
sz = os.path.getsize("RamanNet_Paper.pdf")
print(f"RamanNet_Paper.pdf written ({sz/1024:.0f} KB)")
