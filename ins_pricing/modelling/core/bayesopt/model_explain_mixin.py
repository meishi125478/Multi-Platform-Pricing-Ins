from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import statsmodels.api as sm

try:
    from ...explain import gradients as explain_gradients
    from ...explain import permutation as explain_permutation
    from ...explain import shap_utils as explain_shap
except Exception:  # pragma: no cover - optional for legacy imports
    try:  # best-effort for non-package imports
        from ins_pricing.explain import gradients as explain_gradients
        from ins_pricing.explain import permutation as explain_permutation
        from ins_pricing.explain import shap_utils as explain_shap
    except Exception:  # pragma: no cover
        explain_gradients = None
        explain_permutation = None
        explain_shap = None


class BayesOptExplainMixin:
    def compute_permutation_importance(self,
                                       model_key: str,
                                       on_train: bool = True,
                                       metric: Any = "auto",
                                       n_repeats: int = 5,
                                       max_rows: int = 5000,
                                       random_state: Optional[int] = None):
        if explain_permutation is None:
            raise RuntimeError("explain.permutation is not available.")

        model_key = str(model_key)
        data = self.train_data if on_train else self.test_data
        if self.resp_nme not in data.columns:
            raise RuntimeError("Missing response column for permutation importance.")
        y = data[self.resp_nme]
        w = data[self.weight_nme] if self.weight_nme in data.columns else None

        if model_key == "resn":
            if self.resn_best is None:
                raise RuntimeError("ResNet model not trained.")
            X = self.train_oht_scl_data if on_train else self.test_oht_scl_data
            if X is None:
                raise RuntimeError("Missing standardized features for ResNet.")
            X = X[self.var_nmes]
            predict_fn = lambda df: self.resn_best.predict(df)
        elif model_key == "ft":
            if self.ft_best is None:
                raise RuntimeError("FT model not trained.")
            if str(self.config.ft_role) != "model":
                raise RuntimeError("FT role is not 'model'; FT predictions unavailable.")
            X = data[self.factor_nmes]
            geo_tokens = self.train_geo_tokens if on_train else self.test_geo_tokens
            geo_np = None
            if geo_tokens is not None:
                geo_np = geo_tokens.to_numpy(dtype=np.float32, copy=False)
            predict_fn = lambda df, geo=geo_np: self.ft_best.predict(df, geo_tokens=geo)
        elif model_key == "xgb":
            if self.xgb_best is None:
                raise RuntimeError("XGB model not trained.")
            X = data[self.factor_nmes]
            predict_fn = lambda df: self.xgb_best.predict(df)
        else:
            raise ValueError("Unsupported model_key for permutation importance.")

        return explain_permutation.permutation_importance(
            predict_fn,
            X,
            y,
            sample_weight=w,
            metric=metric,
            task_type=self.task_type,
            n_repeats=n_repeats,
            random_state=random_state,
            max_rows=max_rows,
        )

    # ========= Deep explainability: Integrated Gradients =========

    def compute_integrated_gradients_resn(self,
                                          on_train: bool = True,
                                          baseline: Any = None,
                                          steps: int = 50,
                                          batch_size: int = 256,
                                          target: Optional[int] = None):
        if explain_gradients is None:
            raise RuntimeError("explain.gradients is not available.")
        if self.resn_best is None:
            raise RuntimeError("ResNet model not trained.")
        X = self.train_oht_scl_data if on_train else self.test_oht_scl_data
        if X is None:
            raise RuntimeError("Missing standardized features for ResNet.")
        X = X[self.var_nmes]
        return explain_gradients.resnet_integrated_gradients(
            self.resn_best,
            X,
            baseline=baseline,
            steps=steps,
            batch_size=batch_size,
            target=target,
        )


    def compute_integrated_gradients_ft(self,
                                        on_train: bool = True,
                                        geo_tokens: Optional[np.ndarray] = None,
                                        baseline_num: Any = None,
                                        baseline_geo: Any = None,
                                        steps: int = 50,
                                        batch_size: int = 256,
                                        target: Optional[int] = None):
        if explain_gradients is None:
            raise RuntimeError("explain.gradients is not available.")
        if self.ft_best is None:
            raise RuntimeError("FT model not trained.")
        if str(self.config.ft_role) != "model":
            raise RuntimeError("FT role is not 'model'; FT explanations unavailable.")

        data = self.train_data if on_train else self.test_data
        X = data[self.factor_nmes]

        if geo_tokens is None and getattr(self.ft_best, "num_geo", 0) > 0:
            tokens_df = self.train_geo_tokens if on_train else self.test_geo_tokens
            if tokens_df is not None:
                geo_tokens = tokens_df.to_numpy(dtype=np.float32, copy=False)

        return explain_gradients.ft_integrated_gradients(
            self.ft_best,
            X,
            geo_tokens=geo_tokens,
            baseline_num=baseline_num,
            baseline_geo=baseline_geo,
            steps=steps,
            batch_size=batch_size,
            target=target,
        )

    def _sample_rows(self, data: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(data) == 0:
            return data
        return data.sample(min(len(data), n), random_state=self.rand_seed)

    @staticmethod
    def _shap_nsamples(arr: np.ndarray, max_nsamples: int = 300) -> int:
        min_needed = arr.shape[1] + 2
        return max(min_needed, min(max_nsamples, arr.shape[0] * arr.shape[1]))


    def _build_ft_shap_matrix(self, data: pd.DataFrame) -> np.ndarray:
        matrices = []
        for col in self.factor_nmes:
            s = data[col]
            if col in self.cate_list:
                cats = pd.Categorical(
                    s,
                    categories=self.cat_categories_for_shap[col]
                )
                codes = np.asarray(cats.codes, dtype=np.float64).reshape(-1, 1)
                matrices.append(codes)
            else:
                vals = pd.to_numeric(s, errors="coerce")
                arr = vals.to_numpy(dtype=np.float64, copy=True).reshape(-1, 1)
                matrices.append(arr)
        X_mat = np.concatenate(matrices, axis=1)  # Result shape (N, F)
        return X_mat


    def _decode_ft_shap_matrix_to_df(self, X_mat: np.ndarray) -> pd.DataFrame:
        data_dict = {}
        for j, col in enumerate(self.factor_nmes):
            col_vals = X_mat[:, j]
            if col in self.cate_list:
                cats = self.cat_categories_for_shap[col]
                codes = np.round(col_vals).astype(int)
                codes = np.clip(codes, -1, len(cats) - 1)
                cat_series = pd.Categorical.from_codes(
                    codes,
                    categories=cats
                )
                data_dict[col] = cat_series
            else:
                data_dict[col] = col_vals.astype(float)

        df = pd.DataFrame(data_dict, columns=self.factor_nmes)
        for col in self.cate_list:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df


    def _build_glm_design(self, data: pd.DataFrame) -> pd.DataFrame:
        X = data[self.var_nmes]
        return sm.add_constant(X, has_constant='add')


    def _compute_shap_core(self,
                           model_key: str,
                           n_background: int,
                           n_samples: int,
                           on_train: bool,
                           X_df: pd.DataFrame,
                           prep_fn,
                           predict_fn,
                           cleanup_fn=None):
        if explain_shap is None:
            raise RuntimeError("explain.shap_utils is not available.")
        return explain_shap.compute_shap_core(
            self,
            model_key,
            n_background,
            n_samples,
            on_train,
            X_df=X_df,
            prep_fn=prep_fn,
            predict_fn=predict_fn,
            cleanup_fn=cleanup_fn,
        )

    # ========= GLM SHAP explainability =========

    def compute_shap_glm(self, n_background: int = 500,
                         n_samples: int = 200,
                         on_train: bool = True):
        if explain_shap is None:
            raise RuntimeError("explain.shap_utils is not available.")
        self.shap_glm = explain_shap.compute_shap_glm(
            self,
            n_background=n_background,
            n_samples=n_samples,
            on_train=on_train,
        )
        return self.shap_glm

    # ========= XGBoost SHAP explainability =========

    def compute_shap_xgb(self, n_background: int = 500,
                         n_samples: int = 200,
                         on_train: bool = True):
        if explain_shap is None:
            raise RuntimeError("explain.shap_utils is not available.")
        self.shap_xgb = explain_shap.compute_shap_xgb(
            self,
            n_background=n_background,
            n_samples=n_samples,
            on_train=on_train,
        )
        return self.shap_xgb

    # ========= ResNet SHAP explainability =========

    def _resn_predict_wrapper(self, X_np):
        model = self.resn_best.resnet.to("cpu")
        with torch.no_grad():
            X_tensor = torch.tensor(X_np, dtype=torch.float32)
            y_pred = model(X_tensor).cpu().numpy()
        y_pred = np.clip(y_pred, 1e-6, None)
        return y_pred.reshape(-1)


    def compute_shap_resn(self, n_background: int = 500,
                          n_samples: int = 200,
                          on_train: bool = True):
        if explain_shap is None:
            raise RuntimeError("explain.shap_utils is not available.")
        self.shap_resn = explain_shap.compute_shap_resn(
            self,
            n_background=n_background,
            n_samples=n_samples,
            on_train=on_train,
        )
        return self.shap_resn

    # ========= FT-Transformer SHAP explainability =========

    def _ft_shap_predict_wrapper(self, X_mat: np.ndarray) -> np.ndarray:
        df_input = self._decode_ft_shap_matrix_to_df(X_mat)
        y_pred = self.ft_best.predict(df_input)
        return np.asarray(y_pred, dtype=np.float64).reshape(-1)


    def compute_shap_ft(self, n_background: int = 500,
                        n_samples: int = 200,
                        on_train: bool = True):
        if explain_shap is None:
            raise RuntimeError("explain.shap_utils is not available.")
        self.shap_ft = explain_shap.compute_shap_ft(
            self,
            n_background=n_background,
            n_samples=n_samples,
            on_train=on_train,
        )
        return self.shap_ft
