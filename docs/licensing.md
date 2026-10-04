# Licensing and source-data rights

> **Restricted source datasets must only be ingested when the operator has appropriate
> rights/licence.** This project is a framework: it holds no classification content, and
> running it on a source does not create any right to use that source.

This document is a technical licensing-risk description. It is not legal advice.

## 1. What the project ships

| Item | Content | Status |
|---|---|---|
| Source code | Ingestion, retrieval, validation, API | Project code |
| `app/synthetic/` | Invented, ICD-*like* test classification (`SYNTH-ICD`) and evaluation cases | Original test data, written for this project; it contains no third-party content |
| `data/sources/` | Operator-supplied source files | **Git-ignored. Never committed.** |
| `data/synthetic/` | Regenerated synthetic files | Git-ignored (regenerate with `python -m app.synthetic.cli build`) |

## 2. ICD-10-CA 2022 (CIHI): audit summary

The operator obtained `ICD10CA_2022_final.pdf` together with a one-page French-language
*Conditions d'utilisation de la CIM-10-CA/CCI en format PDF* (Terms of Use, issued by CIHI/ICIS).
Paraphrased, those terms:

- grant **read-only access** for **non-commercial purposes related to the Canadian health
  system**;
- prohibit reproducing ICD-10-CA/CCI in whole or in part, for any purpose;
- prohibit modifying, transforming, copying, distributing, selling or sharing it, **integrating
  it into software**, or creating **derivative works**;
- allow consulting it for personal or internal use only;
- prohibit circumventing mechanisms that restrict the files' security or functionality;
- are superseded by a **separate licence agreement with CIHI**, where one exists.

The PDF itself is encrypted (`/P -3392`): copying/extraction, printing, modification and
annotation are disabled, and only *extraction for accessibility* is allowed.

**Consequence:** that PDF may not be extracted, parsed, stored, indexed or embedded by this
system. The PDF adapter enforces this. It refuses any encrypted PDF whose copy/extract permission
is off, and it never treats the accessibility permission as permission to ingest. The optional OCR
path applies the same check, because rendering pages to OCR them would defeat the same
restriction. A run against the real file confirms the refusal
(`docs/runtime-verification.md` §4): **0 records extracted, `SOURCE_RESTRICTED`.**

**To ingest ICD-10-CA**, obtain the classification under a CIHI licence that covers software and
database use, ideally as CIHI's machine-readable data files, and record the licence basis at
ingestion (§4). The CSV/TSV/XLSX/XML (incl. ClaML)/JSON adapters are designed for exactly that.

### Canadian Coding Standards 2022 (CIHI)

The copyright page of the standards allows **unaltered** reproduction for **non-commercial**
purposes, with CIHI acknowledged as copyright owner. Commercial use needs CIHI's written
authorization. These standards are **not** ingested by this system. Whether structured
segmentation counts as "unaltered" is a question for CIHI or legal counsel.

## 3. Design rules derived from the terms

| Rule | Where enforced |
|---|---|
| Never bypass PDF encryption/permissions; never auto-OCR | `app/ingestion/text/pdf_reader.py`, `layout_adapters.py` |
| Ingestion requires a recorded licence basis | `app/ingestion/cli.py` (`--licence-basis` or `dataset.licence.basis`) |
| Licence statement stored with the dataset | `icd_datasets.metadata.licence`; returned by `GET /datasets/{id}` |
| Content is never sent to a remote embedding provider unless the dataset's metadata allows remote processing | `SearchIndexer.embed_documents` → `LicenceRestrictionError` |
| No bulk export / download / upload endpoint | Route contract test `tests/unit/test_app.py` |
| Admin operations are token-protected and off by default | `app/api/deps.py` |
| Source files and generated data are git-ignored | `.gitignore`, `.dockerignore` |

What the API exposes (codes, titles, rules and provenance, record by record, for a request) is
itself "displaying content to end users". Whether a given deployment may do that, and to whom,
depends on the operator's licence. Deploy the API only to the audiences your licence covers.

## 4. Recording a licence basis

```json
{"dataset": {"coding_system": "ICD-10-CA", "version": "2022", "country": "CA",
             "language": "en", "publisher": "Canadian Institute for Health Information",
             "licence": {"basis": "CIHI licence agreement #..., internal use",
                         "allow_remote_processing": false}}}
```

`ingest` exits with code 2 if no basis is recorded. Set `allow_remote_processing: true` only if
the licence permits sending the content to the configured third-party embedding service.

## 5. Open questions requiring legal confirmation

1. Whether the operator holds, or will obtain, a CIHI licence covering software/database
   integration, AI/embedding use, API exposure and the intended (commercial or non-commercial)
   deployment.
2. Whether the Canadian Coding Standards may be segmented and stored (only "unaltered"
   reproduction is allowed), and whether the intended use is non-commercial.
3. Whether any third-party embedding provider may receive licensed content.
