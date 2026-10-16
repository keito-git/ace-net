"""
DeBERTa-v3-base emotion classifier (text-only supervised baseline).

Backbone: microsoft/deberta-v3-base.
Reference: He et al. (2021). DeBERTaV3: Improving DeBERTa using ELECTRA-Style
  Pre-Training with Gradient-Disentangled Embedding Sharing. ICLR 2023.
  https://arxiv.org/abs/2111.09543

Supports single-label (CrossEntropyLoss) and multi-label (BCEWithLogitsLoss)
classification. This is the primary comparison baseline for ACE-Net.
"""
from typing import Optional

import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class DeBERTaEmotionClassifier(DebertaV2PreTrainedModel):
    """
    DeBERTa-v3-base with a [CLS]-pooling classification head.

    Architecture: DeBERTa-v3-base -> [CLS] hidden state -> Linear + Tanh
    pooler -> Dropout -> Linear classification head. Loss is
    BCEWithLogitsLoss when config.problem_type is "multi_label_classification",
    otherwise CrossEntropyLoss.

    Args:
        config: HuggingFace config (must set num_labels and problem_type)
        dropout_prob: dropout rate applied after the pooler (default 0.1)
    """

    def __init__(self, config, dropout_prob: float = 0.1) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> SequenceClassifierOutput:
        """
        Returns a SequenceClassifierOutput with loss (if labels provided),
        logits, hidden_states, and attentions.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        cls_output = outputs.last_hidden_state[:, 0, :]
        pooled = torch.tanh(self.pooler(cls_output))
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            if self.config.problem_type == "multi_label_classification":
                loss = nn.BCEWithLogitsLoss()(logits, labels.float())
            else:
                loss = nn.CrossEntropyLoss()(logits, labels.long())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
