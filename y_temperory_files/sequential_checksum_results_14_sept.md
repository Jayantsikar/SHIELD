# Checksum Defense Results (Sequential Block Checksum)

Below are the final evaluation metrics for the ResNet-32 ensemble model using the Sequential Block Checksum & Early-Exit Truncation defense strategy against Targeted Bit Trojan (TBT) attacks.

## SUMMARY — Non-Adaptive Ensemble
============================================================
*   **Clean Acc  (no defense)**   : `90.01%`  <- model utility before attack detected
*   **Clean Acc  (defended)**     : `89.93%`  <- accuracy after checksum fires on clean data
*   **ASR        (no defense)**   : `15.48%`  <- raw TBT attack strength
*   **ASR        (defended)**     : `11.67%`  <- after checksum blocks model
------------------------------------------------------------

## SUMMARY — Adaptive Ensemble
============================================================
*   **Clean Acc  (no defense)**   : `87.39%`  <- model utility before attack detected
*   **Clean Acc  (defended)**     : `83.02%`  <- accuracy after checksum fires on clean data
*   **ASR        (no defense)**   : `40.72%`  <- raw TBT attack strength
*   **ASR        (defended)**     : `15.09%`  <- after checksum blocks model
------------------------------------------------------------
