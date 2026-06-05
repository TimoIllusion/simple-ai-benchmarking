# Project Name: simple-ai-benchmarking
# File Name: generation_factory.py
# Author: Timo Leitritz
# Copyright (C) 2024 Timo Leitritz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from simple_ai_benchmarking.config_structures import GenerationModelConfig


class GenerationModelFactory:
    """Builds local generative models for generation workloads.

    Kept separate from ClassificationModelFactory because a language model is not
    a classifier (no num_classes / image input shape)."""

    @staticmethod
    def create_pytorch_model(model_cfg: GenerationModelConfig):
        from simple_ai_benchmarking.models.pt.simple_transformer_lm import (
            SimpleTransformerLanguageModel,
        )

        return SimpleTransformerLanguageModel(
            vocab_size=model_cfg.vocab_size,
            context_length=model_cfg.context_length,
            embedding_dim=model_cfg.embedding_dim,
            num_heads=model_cfg.attention_heads,
            num_layers=model_cfg.transformer_layers,
            feedforward_dim=model_cfg.feedforward_dim,
        )
