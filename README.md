# Bioinfo_Cancer-subtyping

This repository contains all the necessary code to rebuild the DeepCC and MoGCN models with newer data.

### Data Flow for DeepCC:
- Stage One: Data Preparation
  - Patients' data, including mRNAseq and clinical data
  - Human Gene sets from MsigDB

- Stage Two: Data Engineering
  - Normalization of patients' data (converted to TPM and then log2-transformed/MAS5.0)
  - Calculation of a list of Enrichment Scores using GSEA, with patients' genomic data and Human Gene sets from MsigDB as input

- Stage Three: Model Training & Validation
  - Construction of the DeepCC Neural Network and performing training/validation

- Stage Four: Encapsulation
  - Building a single-sample predictor
  - Developing evaluation metrics
  - Implementing Functional analysis and visualization

### Data Flow for MoGCN:
- Stage One: Data Preparation
  - Patients' data, including CNV, RNA-seq at the transcriptomic level, RPPA, and clinical data
- Stage Two: Data Transformation
  - Utilizing Encoder/Decoder to convert CNV, RNA-seq at the transcriptomic level, RPPA, and clinical data into features
  - Employing Similarity Network Fusion to integrate different types of omics data and create a Patients' similarity network
- Stage Three: GCN
  - Taking features and graphs as input and building another GCN model.