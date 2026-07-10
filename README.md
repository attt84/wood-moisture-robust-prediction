# 木材含水率のロバスト予測AI（近赤外分光スペクトル）

本リポジトリは、近赤外分光スペクトル（波長データ）から木材の含水率を非破壊で予測するAIモデル構築プロジェクトのポートフォリオです。

> ⚠️ **注意**: コンペティションの「情報公開ポリシー」に準拠し、本パブリックリポジトリには実行可能なソースコードおよびデータセットの実体は含めておりません。解法アプローチとコアロジックのスニペットのみを公開しています。

---

## 1. 概要と最終的なアーキテクチャ（結論）

*   **目的**: 近赤外分光スペクトル（波長データ）から木材の含水率を予測するAIモデルの構築。
*   **課題（ドメインシフト）**: 学習データと評価データで「対象の樹種」が完全に異なる設定。既知の樹種に過学習せず、未知の樹種でも破綻しない「汎化性能」が強く求められました。
*   **最終的な実装内容（Feature Dropout Hybrid CNN）**:
    生波形（SG平滑化）からテクスチャを抽出する1D-CNNと、物理・分光学の知見から設計した「手作り特徴量（マクロな物理量）」をネットワークの深層（Late Fusion）で結合させるハイブリッドモデルを構築。
    さらに、手作り特徴量への過学習（ショートカット学習）を防ぐため、物理特徴量に対して意図的に情報を欠落させる「Feature Dropout (20%)」を導入し、未知の樹種に対する極めて高い汎化性能を獲得しました。

### コアロジックの実装スニペット
ショートカット学習を打破し、最強の汎化性能を獲得したアーキテクチャ（PyTorch実装）です。

```python
import torch
import torch.nn as nn

class RobustHybridCNN(nn.Module):
    def __init__(self, use_dropout=True):
        super().__init__()
        # 1D-CNN for Waveform Texture (SG Smoothed)
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(16), nn.ReLU(), nn.AvgPool1d(2),
            nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.AvgPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(16)
        )
        self.flatten = nn.Flatten()
        
        # Late Fusion at Deep Layer (64-dim)
        self.fc_cnn = nn.Linear(64 * 16, 64)
        self.drop_cnn = nn.Dropout(0.3)
        self.act = nn.ReLU()
        
        # ⚠️ Breakthrough: Feature Dropout for Physics Features
        self.use_dropout = use_dropout
        self.feat_dropout = nn.Dropout(0.2) # Forces CNN to learn waveform instead of memorizing IDs
        
        # Fused Predictor
        self.fc_fused = nn.Linear(64 + 7, 32)
        self.drop_fused = nn.Dropout(0.2)
        self.out = nn.Linear(32, 1)

    def forward(self, x_seq, x_phys):
        # Process Waveform
        h = self.conv_blocks(x_seq)
        h = self.flatten(h)
        h = self.act(self.drop_cnn(self.fc_cnn(h)))
        
        # Apply Feature Dropout to Hand-crafted Physics Features
        if self.use_dropout and self.training:
            x_phys = self.feat_dropout(x_phys)
            
        # Concat and Predict
        h_fused = torch.cat([h, x_phys], dim=1)
        h_fused = self.act(self.drop_fused(self.fc_fused(h_fused)))
        return self.out(h_fused)
```

---

## 2. 実験過程とRMSE（精度）の推移

コンペティション期間中、アーキテクチャの改善による「未知の樹種（特に難解なSp3, Sp11）」に対するCross-Validation（Leave-One-Group-Out）のRMSE推移は以下の通りです。

| フェーズ | アプローチ概要 | Overall CV (RMSE) | 難解樹種 (Sp3 / Sp11) のRMSE | 状態・評価 |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 45** | ピュア1D-CNN（波形のみ学習） | 約 10.00 | 12.00前後 / 11.50前後 | ドメインシフトに対して堅牢。ベースラインとなるスコア。 |
| **Phase 98** | CNN ＋ 物理特徴量の追加 | 8.70 | 13.90 / 12.50 | 全体CVは見かけ上向上したが、未知樹種で崩壊（過学習によるショートカットの発生）。 |
| **Phase 99** | Phase 98 ＋ **Feature Dropout (20%)** | **8.36** | **9.93** / **9.71** | 特効薬として機能し、難解な未知樹種に対しても「10.0」を下回る最高の汎化性能を達成。 |

---

## 3. 実験中の重要論点と、決断の根拠

本プロジェクトでは、単にスコアを競うだけでなく、「モデルが何を学習しているのか」を物理学的に深掘りし、数々の重要な決断を行いました。

### 論点A：手作り特徴量は「汎化」を助けるか、妨げるか？
*   **課題**: 水分の吸収ピーク（A5150等）を直接特徴量化してCNNに与えたところ、かえって未知の樹種に対する精度が悪化してしまいました（Phase 98）。
*   **分析と決断**: EDAを通じ、特定の樹種においては「含水率」と「A5150の高さ」に強い見せかけの相関（スプリアス相関）があることを発見。モデルは波形の「テクスチャ」を理解するのをやめ、手作り特徴量を「樹種のID（バーコード）」として丸暗記（ショートカット学習）していました。
*   **根拠**: この暗記を強制排除するため、物理特徴量側にだけ **Feature Dropout (20%)** を適用。結果として、モデルはサボらずに波形のテクスチャを学習するようになり、未知樹種でのRMSEが劇的に改善しました。

![Feature Dropoutの効果](portfolio_assets/14_eda_species_bias_stable.png)
*(図: 物理特徴量に対する強いバイアス。AIは波形ではなく「特定の高さ＝特定の樹種」というショートカットを学習してしまっていたため、特徴量を強制的にDropさせる決断を行った)*

### 論点B：プーリング層の選択（MaxPool vs AvgPool）
*   **課題**: ベースラインのCNNでは一般的な `MaxPool1d` を使用していましたが、波形データにおいてそれが最適かが議論になりました。
*   **分析と決断**: 画像認識の輪郭抽出と異なり、分光スペクトルにおける情報（水分子の量）は波形の「面積」や「エネルギー量」に依存します。そのため、スパイクノイズを拾いやすいMaxPoolではなく、面積情報を保存する `AvgPool1d` に全てのプーリング層を統一する決断をしました。
*   **根拠**: 吸光度の「総量保存」という物理学的な観点からのアプローチであり、実際に予測の安定性（ロバスト性）向上に大きく寄与しました。

![スペクトルの樹種間差異](portfolio_assets/01_spectral_evolution_per_species.png)
*(図: 樹種ごとの近赤外分光スペクトルの違い。同じ含水率でも、樹種（細胞構造や密度）が異なるとベースラインや散乱状態が大きく異なる)*

### 論点C：物理特徴量との融合位置（Early Fusion vs Late Fusion）
*   **課題**: CNNによる波形特徴抽出の「どの段階」で物理特徴量と混ぜ合わせるかが論点となりました。
*   **分析と決断**: ネットワークの浅い層（Early Fusion）で混ぜると物理特徴量が薄まりすぎ、逆に深すぎる位置で混ぜると物理特徴量に依存しすぎるというジレンマがありました。
*   **根拠**: Ablation Study（段階的な比較実験）の結果、CNNの特徴を64次元まで圧縮し、十分に熟成された段階で7次元の物理特徴量と結合させる「Late Fusion」が、波形のテクスチャとマクロな物理量のバランスを最も良く保つことが定量的に証明されたため、最終アーキテクチャに採用しました。
