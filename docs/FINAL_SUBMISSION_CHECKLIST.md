# Official Solar 2026 final submission: tasks requiring account/identity

**Status:** all locally executable inference and numerical checks listed below have been completed. External account publication and organizer scoring have NOT been done. Do not confuse a ZIP with the actual Kaggle prediction upload.

1. **Kaggle competition:** Join / confirm accepted competition rules on your account at https://www.kaggle.com/competitions/filament-segmentation-2026 . Upload the **plain `submission.csv`** through the competition submission interface. Check for accepted submission status and official leaderboard PQ before asserting any score. The user/participant controls Kaggle account credentials.
2. **Git-based repository:** Push the contents of this source directory to an accessible GitHub / GitLab / Bitbucket repository. Include bundled `results/SOLAR_REAL_TRAINED_UNET.pt` (essential to reproduce the CSV), both notebook variants, `requirements.txt` and source. Never commit `00.zip`, the original `MAGFiLO_1.0_Kaggle_2026.zip`, passwords, API tokens or proprietary material. Verify repository accessibility without authentication.
3. **Four-page report:** Review `report/technical_report_4pages.pdf` against the *exact* organizer Overleaf template linked in the [Final Submission section](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/final-submission); the supplied source uses ACM `acmart/sigconf` but the official Overleaf template source is not independently downloadable here. Replace author/team and affiliation placeholders in `report/main.tex`, rebuild with `pdflatex` and confirm exactly four pages. The submitted PDF should be the reviewed one.
4. **Google Form:** Open the organizer's Google Form from the Kaggle competition overview's Final Submission section, add the final report and public Git URL, supply accurate participant details and submit by **Nov 15, 2026, 11:59 p.m. CDT**. Retain confirmation. The raw form URL is not invented here.
5. **After filing:** Select final Kaggle submission(s) if required; monitor organizer notices and maintain the public repository until after winner announcement.

Engineering evidence available here:

- Fully regenerated 180-image test CSV from original authorized JPEGs; 1,013 RLE rows; SHA-256 `e0e55cf4eb383769c17b742d556d619859307b82a4267792ad2345698126d9ea`.
- Executed 14-cell notebook with no hidden-data access; `results/SOLAR_NOTEBOOK_EXECUTION_RECEIPT.json`.
- Four-page ACM-format report, saved TeX and figures, local actual 2048px PQ 0.273115 with paired error distribution.
- `scripts/final_preflight.py` and `tests/` run without needing to retrain the model.

**Remaining human-specific information:** participant's chosen public Git URL, name/affiliation/eligibility and the actual Kaggle + Google Form confirmation. No false claim that these have already happened.
