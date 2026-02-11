import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io

class ForwardCurveEngine:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.load_data()
        self.tenors = self.df.columns.tolist()

    def load_data(self):
        # Đọc dữ liệu
        df = pd.read_excel(self.file_path, index_col=0)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Chuyển sang số và xử lý lỗi
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Xử lý NaN: Fill forward rồi backward, cuối cùng điền 0 nếu vẫn lỗi
        df = df.ffill().bfill().fillna(0)
        return df

    def to_excel(self):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            self.df.to_excel(writer, sheet_name='LME_Data')
        return output.getvalue()

    def get_basic_stats(self, tenor):
        # Nếu kỳ hạn không tồn tại
        if tenor not in self.df.columns:
            return {
                "Mean": 0.0, "Annual Vol": 0.0, "Skewness": 0.0, "Kurtosis": 0.0,
                "Min": 0.0, "Max": 0.0, "Returns": pd.Series(dtype=float)
            }
        
        series = self.df[tenor]
        
        # Tính Returns và làm sạch triệt để (Bỏ vô cực, bỏ NaN)
        returns = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        
        # Trả về Dictionary có chứa key "Returns"
        return {
            "Mean": float(series.mean()),
            "Annual Vol": float(returns.std() * np.sqrt(252)) if not returns.empty else 0.0,
            "Skewness": float(skew(returns)) if len(returns) > 2 else 0.0,
            "Kurtosis": float(kurtosis(returns)) if len(returns) > 2 else 0.0,
            "Min": float(series.min()),
            "Max": float(series.max()),
            "Returns": returns  # <--- DÒNG QUAN TRỌNG ĐỂ KHÔNG BỊ LỖI KEYERROR
        }

    def run_pca_analysis(self, n_components=3):
        # Tính returns
        ret = self.df.pct_change()
        # Loại bỏ các dòng chứa NaN hoặc Inf (bắt buộc cho PCA)
        ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
        
        if ret.empty:
            # Trả về dữ liệu rỗng nếu không tính được để app không sập
            return [0, 0, 0], pd.DataFrame()

        scaler = StandardScaler()
        scaled = scaler.fit_transform(ret)
        
        pca = PCA(n_components=n_components)
        pca.fit(scaled)
        
        comp_df = pd.DataFrame(
            pca.components_.T, 
            index=ret.columns, 
            columns=['Level (PC1)', 'Slope (PC2)', 'Curvature (PC3)']
        )
        return pca.explained_variance_ratio_.tolist(), comp_df

    def get_correlation_matrix(self):
        # Làm sạch dữ liệu trước khi tính Correlation
        return self.df.pct_change().replace([np.inf, -np.inf], np.nan).dropna().corr()