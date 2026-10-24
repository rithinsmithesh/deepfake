"""
Multimodal Trust & Integration Engine for DeepShield
Aggregates genuinely computed visual predictions and audio-lip synchronization signals.
Adheres strictly to scientific transparency: never invents scores when components are unavailable.
"""


class DeepShieldTrustEngine:
    def __init__(self, visual_weight=0.75, sync_weight=0.25):
        self.visual_weight = visual_weight
        self.sync_weight = sync_weight

    def evaluate(self, visual_result, audio_result, sync_result, visual_weight=None, sync_weight=None, *args, **kwargs):
        """
        Evaluates and combines genuine analysis outputs into an exact multimodal verdict score.
        """
        if visual_weight is None:
            visual_weight = kwargs.get("visual_weight", None)
        if sync_weight is None:
            sync_weight = kwargs.get("sync_weight", None)
        if len(args) >= 1 and visual_weight is None:
            visual_weight = args[0]
        if len(args) >= 2 and sync_weight is None:
            sync_weight = args[1]

        w_vis = visual_weight if visual_weight is not None else self.visual_weight
        w_sync = sync_weight if sync_weight is not None else self.sync_weight

        # Normalize weights if sum != 1.0
        total_w = w_vis + w_sync
        if total_w > 0:
            w_vis = w_vis / total_w
            w_sync = w_sync / total_w

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
                "deepfake_probability": None,
                "authenticity_label": "Inconclusive (Visual Checkpoint Missing)",
                "confidence_level": "N/A",
                "forensic_grade": "INCONCLUSIVE",
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
        vis_real_prob = float(vis_details["mean_real_prob"])  # 0 to 100
        vis_fake_prob = float(vis_details["mean_fake_prob"])

        # Subcase A: Audio and Lip Sync are both available
        if components["sync"]["available"]:
            sync_details = sync_result
            sync_score = float(sync_details["sync_score"])  # 0 to 100
            # Scale sync score to a 0-100 authenticity scale based on FakeAVCeleb empirical real-world distribution
            # In FakeAVCeleb, typical real videos scored between 5 and 25
            normalized_sync_authenticity = min(100.0, (sync_score / 25.0) * 100.0)

            vis_pts = round(w_vis * vis_real_prob, 2)
            sync_pts = round(w_sync * normalized_sync_authenticity, 2)

            trust_score = round(max(0.0, min(100.0, vis_pts + sync_pts)), 2)
            deepfake_probability = round(max(0.0, min(100.0, 100.0 - trust_score)), 2)

            if trust_score >= 65.0:
                auth_label = "Verified Authentic (Human)"
                verdict_color = "emerald"
                forensic_grade = "GRADE A (Authentic Human)" if trust_score >= 80.0 else "GRADE B (Likely Authentic)"
            elif trust_score >= 40.0:
                auth_label = "Suspicious / Inconclusive Anomaly"
                verdict_color = "amber"
                forensic_grade = "GRADE C (Inconclusive Anomaly)"
            else:
                auth_label = "Synthetic Deepfake Detected"
                verdict_color = "rose"
                forensic_grade = "GRADE F (Deepfake Tampered)"

            # Confidence level calculation
            dist_from_mid = abs(trust_score - 50.0)
            if dist_from_mid >= 25.0:
                confidence_rating = f"HIGH ({round(50.0 + dist_from_mid, 1)}% certainty)"
            elif dist_from_mid >= 12.0:
                confidence_rating = f"MODERATE ({round(50.0 + dist_from_mid, 1)}% certainty)"
            else:
                confidence_rating = f"MARGINAL ({round(50.0 + dist_from_mid, 1)}% boundary)"

            reasoning = (
                f"Multimodal Fusion: Spatial CNN authenticity is {vis_real_prob:.1f}% (+{vis_pts:.1f} pts at {round(w_vis*100)}% weight). "
                f"Audio-Lip sync score is {sync_score:.1f}/100 (normalized temporal coherence: {normalized_sync_authenticity:.1f}%, +{sync_pts:.1f} pts at {round(w_sync*100)}% weight). "
                f"Exact Multimodal Trust Score: {trust_score:.2f}/100 | Deepfake Manipulation Risk: {deepfake_probability:.2f}%."
            )

        # Subcase B: No audio or synchronization failed
        else:
            vis_pts = round(w_vis * vis_real_prob, 2)
            sync_pts = 0.0
            normalized_sync_authenticity = 0.0
            sync_score = 0.0

            trust_score = round(max(0.0, min(100.0, vis_pts)), 2)
            deepfake_probability = round(max(0.0, min(100.0, 100.0 - trust_score)), 2)

            if trust_score >= 45.0:
                auth_label = "Inconclusive / Degraded (Audio Track Missing)"
                verdict_color = "amber"
                forensic_grade = "GRADE C- (Audio Modality Missing)"
            else:
                auth_label = "Likely Manipulated (Visual Flags & Missing Audio)"
                verdict_color = "rose"
                forensic_grade = "GRADE F (Tampered & Missing Audio)"

            confidence_rating = "DEGRADED (Single Modality)"
            sync_reason = components["sync"]["details"].get("reason") if components["sync"]["details"] else "No audio stream detected."
            reasoning = (
                f"Degraded Multimodal Assessment: Video lacks valid audio synchronization ({sync_reason}). "
                f"Temporal trust weight is nullified. Exact Trust Score is strictly the weighted visual evidence: "
                f"{vis_real_prob:.1f}% × {w_vis:.2f} = {trust_score:.2f} / 100 | Deepfake Risk: {deepfake_probability:.2f}%."
            )

        return {
            "status": "computed",
            "trust_score": trust_score,
            "deepfake_probability": deepfake_probability,
            "authenticity_label": auth_label,
            "verdict_color": verdict_color,
            "forensic_grade": forensic_grade,
            "confidence_rating": confidence_rating,
            "weights_used": {"visual": round(w_vis, 2), "sync": round(w_sync, 2)},
            "points_breakdown": {
                "visual_points": vis_pts,
                "sync_points": sync_pts,
                "visual_real_prob": vis_real_prob,
                "visual_fake_prob": vis_fake_prob,
                "sync_raw_score": sync_score,
                "sync_normalized": normalized_sync_authenticity
            },
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
