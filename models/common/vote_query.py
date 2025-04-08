import torch
import torch.nn as nn

from libs.pc_utils import farthest_point_sample
from models.common.pointnet2_modules import PointnetSAModuleVotes


class VoteQuery(nn.Module):
    def __init__(self, d_model, nqueries):
        super().__init__()
        self.nqueries = nqueries

        self.FFN_vote = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, 3 + d_model, 1),
        )

        self.set_abstraction = PointnetSAModuleVotes(
            npoint=nqueries,
            radius=0.3,
            nsample=16,
            mlp=[d_model, d_model, d_model, d_model],
            use_xyz=True,
            normalize_xyz=True
        )

    def forward(self, encode_xyz, encode_features):
        """
        Forward pass.
        Arguments:
            encode_xyz: (batch_size, num_seed, 3) Pytorch tensor
            encode_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, vote_feature_dim, num_seed*vote_factor)
        """

        # batch, channel, npoints
        out = self.FFN_vote(encode_features)
        vote_xyz = encode_xyz + out[:, :3, :].permute(0, 2, 1)
        encode_features = encode_features + out[:, 3:, :]

        features_norm = torch.norm(encode_features, p=2, dim=1)
        encode_features = encode_features.div(features_norm.unsqueeze(1))

        sample_inds = farthest_point_sample(encode_xyz, self.nqueries)

        query_xyz, query_features, _ = self.set_abstraction(
            vote_xyz,
            encode_features,
            sample_inds
        )

        return {
            'vote_xyz': vote_xyz,  # batch x npenc x 3
            'seed_xyz': encode_xyz,  # batch x npenc x 3
            'query_xyz': query_xyz,  # batch x npenc x 3
            'query_features': query_features  # batch x channel x npquery
        }
