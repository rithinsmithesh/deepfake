"""
Multimodal Trust & Integration Engine for DeepShield
Aggregates genuinely computed visual predictions and audio-lip synchronization signals.
Adheres strictly to scientific transparency: never invents scores when components are unavailable.
"""


class DeepShieldTrustEngine:
    def __init__(self, visual_weight=0.75, sync_weight=0.25):
        self.visual_weight = visual_weight
        self.sync_weight = sync_weight

    def evaluate(self, visual_result, audio_result, sync_result):
        """
        Evaluates and combines genuine analysis outputs.
        """
        components = {
            "visual": {
                "available": visual_result is not None and visual_result.get("status") == "success",
                "details": visual_result
            },
            "audio": {
                "available": audio_result is not None and audio_result.get("has_audio", False),
                "details": audio_result
            },
            "sync": {
                "available": sync_result is not None and sync_result.get("success", False),
                "details": sync_result
            }
        }

        # Case 1: Visual model checkpoint is NOT loaded
        if not components["visual"]["available"]:
            return {
                "status": "partial_signals",
                "trust_score": None,
                "authenticity_label": "Inconclusive (Visual Checkpoint Missing)",
                "confidence_level": "N/A",
                "components": components,
                "reasoning": (
                    "A preliminary Trust Score cannot be ethically calculated without visual model weights. "
                    "EfficientNet-B0 inference provides the primary spatial deepfake detection signal (75% weight). "
                    "Place 'FINAL_EfficientNet_B0_FakeAVCeleb.pth' in models/ to enable full multimodal fusion."
                ),
                "formula_definition": self._get_formula_explanation()
            }

        # Case 2: Visual model is available
        vis_details = visual_result
        vis_real_prob = vis_details["mean_real_prob"]  # 0 to 100

        # Subcase A: Audio and Lip Sync are both available
        if components["sync"]["available"]:
            sync_details = sync_result
            sync_score = sync_details["sync_score"]  # 0 to 100
            # Scale sync score to a 0-100 authenticity scale based on FakeAVCeleb empirical real-world distribution
            # In FakeAVCeleb, typical real videos scored between 5 and 25
            normalized_sync_authenticity = min(100.0, (sync_score / 25.0) * 100.0)

            trust_score = (self.visual_weight * vis_real_prob) + (self.sync_weight * normalized_sync_authenticity)
            trust_score = round(max(0.0, min(100.0, trust_score)), 2)

            if trust_score >= 65.0:
                auth_label = "Likely Authentic"
                verdict_color = "emerald"
            elif trust_score >= 40.0:
                auth_label = "Suspicious / Inconclusive"
                verdict_color = "amber"
            else:
                auth_label = "Likely Manipulated (Deepfake)"
                verdict_color = "rose"

            reasoning = (
                f"Multimodal Fusion: Visual feature authenticity probability is {vis_real_prob:.1f}%, "
                f"and Audio-Lip synchronization score is {sync_score:.1f}/100 "
                f"(normalized temporal consistency: {normalized_sync_authenticity:.1f}%). "
                f"Combined weighted Trust Score is {trust_score:.1f}/100."
            )

        # Subcase B: No audio or synchronization failed
        else:
            # Forensic Integrity: When audio is missing or stripped, cross-modal verification is impossible.
            # We do NOT re-inflate visual weight to 100%. Instead, the missing 25% sync modality contributes 0,
            # which naturally penalizes videos with stripped audio tracks to prevent evasion attacks.
            trust_score = round(self.visual_weight * vis_real_prob, 2)
            
            # Since an entire forensic modality is missing, max verdict is capped at "Inconclusive / Degraded"
            if trust_score >= 45.0:
                auth_label = "Inconclusive / Degraded (Audio Track Missing)"
                verdict_color = "amber"
            else:
                auth_label = "Likely Manipulated (Visual Flags & Missing Audio)"
                verdict_color = "rose"

            sync_reason = components["sync"]["details"].get("reason") if components["sync"]["details"] else "No audio stream detected."
            reasoning = (
                f"Degraded Multimodal Assessment: Video lacks valid audio synchronization ({sync_reason}). "
                f"In forensic deepfake analysis, stripped audio is a potential evasion tactic. "
                f"Therefore, the 25% temporal trust weight is nullified. Trust score is strictly the weighted visual "
                f"evidence: {vis_real_prob:.1f}% × 0.75 = {trust_score:.1f} / 100."
            )

        return {
            "status": "computed",
            "trust_score": trust_score,
            "authenticity_label": auth_label,
            "verdict_color": verdict_color,
            "components": components,
            "reasoning": reasoning,
            "formula_definition": self._get_formula_explanation()
        }

    def _get_formula_explanation(self):
        return {
            "formula": "Trust Score = (0.75 × Visual Authenticity %) + (0.25 × Normalized Sync %)",
            "weights": {
                "visual_spatial": "75% (EfficientNet-B0 CNN per-frame feature classifier)",
                "temporal_sync": "25% (Pearson correlation of RMS speech energy and lip aperture)",
            },
            "normalization_logic": (
                "Empirical FakeAVCeleb baselines show real videos average ~9.4 sync score with maximums around ~30-40. "
                "The raw sync score is normalized against benchmark thresholds (score / 25.0 × 100) before fusion."
            ),
            "disclaimer": (
                "Important Scientific Transparency Notice: The Trust Score is an experimental college research heuristic. "
                "It is designed to illustrate multimodal signal integration and should NOT be treated as a legally certified "
                "forensic certainty or a production-hardened biometric proof."
            )
        }
