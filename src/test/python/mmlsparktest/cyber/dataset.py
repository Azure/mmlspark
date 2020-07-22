# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in project root for information.

import pandas as pd
import random
from typing import Optional
from mmlsparktest.spark import *
from mmlspark.cyber.anomaly import AccessAnomaly
import itertools


class CfAlgoParams:
    """
    Parameter for the AccessAnomaly Estimator indicating
    if to use implicit or explicit feedback versions of the ALS algorithm
    and additional parameters that are relevant for the implicit/explicit versions
    """
    def __init__(self, implicit: bool):
        self._implicit = implicit
        self._alpha: Optional[float] = None
        self._complementset_factor: Optional[int] = None
        self._neg_score: Optional[float] = None

        # set default values
        if implicit:
            self.set_alpha(1.0)
        else:
            self.set_complementset_factor(2)
            self.set_neg_score(1.0)

    @property
    def implicit(self) -> bool:
        return self._implicit

    @property
    def alpha(self) -> Optional[float]:
        """
        Relevant for the implicit version only
        :return: alpha value (see descriptions in explainParam of AccessAnomaly)
        """
        return self._alpha

    @property
    def complementset_factor(self) -> Optional[int]:
        """
        Relevant for the explicit version only
        :return: complementset_factor value (see descriptions in explainParam of AccessAnomaly)
        """
        return self._complementset_factor

    @property
    def neg_score(self) -> Optional[float]:
        """
        Relevant for the explicit version only
        :return: neg_score value (see descriptions in explainParam of AccessAnomaly)
        """
        return self._neg_score

    def set_alpha(self, value: float):
        assert self.implicit
        self._alpha = value
        return self

    def set_complementset_factor(self, value: int):
        assert not self.implicit
        self._complementset_factor = value
        return self

    def set_neg_score(self, value: float):
        assert not self.implicit
        assert value > 0.0
        self._neg_score = value
        return self


class AccessAnomalyConfig:
    """
    Define default values for AccessAnomaly Params
    """
    default_tenant_col = 'tenant'
    default_user_col = 'user'
    default_res_col = 'res'
    default_likelihood_col = 'likelihood'
    default_output_col = 'anomaly_score'

    default_rank = 10
    default_max_iter = 25
    default_reg_param = 1.0
    default_num_blocks = None   # |tenants| if separate_tenants is False else 10
    default_separate_tenants = False

    default_low_value = 5.0
    default_high_value = 10.0

    default_algo_cf_params = CfAlgoParams(True)

    @staticmethod
    def create_access_anomaly(**params):
        aa = AccessAnomaly()

        aa.setTenantCol(params.get('tenant_col', AccessAnomalyConfig.default_tenant_col))
        aa.setUserCol(params.get('user_col', AccessAnomalyConfig.default_user_col))
        aa.setResCol(params.get('res_col', AccessAnomalyConfig.default_res_col))
        aa.setAccessCol(params.get('likelihood_col', AccessAnomalyConfig.default_likelihood_col))
        aa.setOutputCol(params.get('output_col', AccessAnomalyConfig.default_output_col))

        aa.setRank(params.get('rank', AccessAnomalyConfig.default_rank))
        aa.setMaxIter(params.get('max_iter', AccessAnomalyConfig.default_max_iter))
        aa.setReg(params.get('reg_param', AccessAnomalyConfig.default_reg_param))
        aa.setNumBlocks(params.get('num_blocks', 1))
        aa.setSeparateTenants(params.get('separate_tenants', AccessAnomalyConfig.default_separate_tenants))

        return aa


class DataFactory:
    def __init__(self):
        self.hr_users = ['hr_user_' + str(i) for i in range(7)]
        self.hr_resources = ['hr_res_' + str(i) for i in range(30)]

        self.fin_users = ['fin_user_' + str(i) for i in range(5)]
        self.fin_resources = ['fin_res_' + str(i) for i in range(25)]

        self.eng_users = ['eng_user_' + str(i) for i in range(10)]
        self.eng_resources = ['eng_res_' + str(i) for i in range(50)]

        self.rand = random.Random(42)

    def to_pdf(self, users, resources, likelihoods) -> pd.DataFrame:
        return pd.DataFrame(
            data={
                AccessAnomalyConfig.default_user_col: [str(u) for u in users],
                AccessAnomalyConfig.default_res_col: [str(r) for r in resources],
                AccessAnomalyConfig.default_likelihood_col: [float(s) for s in likelihoods]
            }
        )

    def tups2pdf(self, tup_arr):
        user_lst = [tup[0] for tup in tup_arr]
        res_lst = [tup[1] for tup in tup_arr]
        likelihood_lst = [tup[2] for tup in tup_arr]

        return self.to_pdf(user_lst, res_lst, likelihood_lst)

    def edges_between(self, users, resources, ratio, full_node_coverage, not_set=None):
        required_edge_cnt = len(users) * len(resources) * ratio
        tups = []
        seen = set([])
        seen_users = set([])
        seen_resources = set([])

        # optimization for creating dense access patterns (fill all the possible pairs in advance)
        cart = list(itertools.product(range(len(users)), range(len(resources)))) if ratio >= 0.5 else None

        while len(tups) < required_edge_cnt \
                or (full_node_coverage and (len(seen_users) < len(users)) or (len(seen_resources) < len(resources))):

            if cart is not None:
                ii = self.rand.randint(0, len(cart) - 1)
                ui, ri = cart[ii]
                cart[ii] = cart[-1]
                cart.pop()
            else:
                ui = self.rand.randint(0, len(users) - 1)
                ri = self.rand.randint(0, len(resources) - 1)

            user = users[ui]
            res = resources[ri]

            if ((ui, ri) in seen) or ((not_set is not None) and ((user, res) in not_set)):
                continue

            seen.add((ui, ri))
            seen_users.add(ui)
            seen_resources.add(ri)

            assert users[ui] is not None
            assert resources[ri] is not None

            score = self.rand.randint(500, 1000)
            tups.append((user, res, score))

        return tups

    def create_clustered_training_data(self, ratio=0.25):
        return self.tups2pdf(
            self.edges_between(self.hr_users, self.hr_resources, ratio, True) +
            self.edges_between(self.fin_users, self.fin_resources, ratio, True) +
            self.edges_between(self.eng_users, self.eng_resources, ratio, True)
        )

    def create_clustered_intra_test_data(self, train=None):
        not_set = set(
            [(row[AccessAnomalyConfig.default_user_col],
              row[AccessAnomalyConfig.default_res_col]) for _, row in train.iterrows()]
        ) if train is not None else None

        return self.tups2pdf(
            self.edges_between(self.hr_users, self.hr_resources, 0.025, False, not_set) +
            self.edges_between(self.fin_users, self.fin_resources, 0.05, False, not_set) +
            self.edges_between(self.eng_users, self.eng_resources, 0.035, False, not_set)
        )

    def create_clustered_inter_test_data(self):
        return self.tups2pdf(
            self.edges_between(self.hr_users, self.fin_resources, 0.025, False) +
            self.edges_between(self.hr_users, self.eng_resources, 0.025, False) +
            self.edges_between(self.fin_users, self.hr_resources, 0.05, False) +
            self.edges_between(self.fin_users, self.eng_resources, 0.05, False) +
            self.edges_between(self.eng_users, self.fin_resources, 0.035, False) +
            self.edges_between(self.eng_users, self.hr_resources, 0.035, False)
        )

    def create_fixed_training_data(self):
        users = [
            6, 2, 8, 6, 7, 8, 2, 8, 10, 3, 11, 6, 7, 3, 2, 10, 11, 1, 8, 4, 9, 3, 5, 6, 7
        ]

        resources = [
            2, 8, 2, 8, 2, 6, 4, 1, 5, 5, 6, 1, 6, 7, 6, 3, 3, 4, 4, 8, 2, 1, 7, 7, 5
        ]

        likelihoods = [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            46.77678298950195,
            35.94552993774414,
            36.470367431640625,
            33.59884262084961,
            10.0,
            43.84599304199219,
            14.908903121948242,
            10.0,
            19.817806243896484,
            21.398120880126953,
            14.908903121948242
        ]

        return self.to_pdf(users, resources, likelihoods)
