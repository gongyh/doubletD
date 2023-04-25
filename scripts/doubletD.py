#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 2020

@author: Palash Sashittal
"""

import pandas as pd
import sys
import argparse
import itertools
import math
import numpy as np

class doubletFinder():

    def __init__(self, df_total, df_alt, delta, beta, mu_hetero, mu_hom, alpha_fp, alpha_fn, missing=False, verbose=True, binom=False, precision=None):

        self.df_total = df_total
        self.df_alt = df_alt
        self.delta = delta
        self.beta = beta
        self.mu_hetero = mu_hetero
        self.mu_hom = mu_hom
        self.alpha_fp = alpha_fp
        self.alpha_fn = alpha_fn
        self.precision = precision
        self.missing = missing
       
        if binom:
            # prv_y符合二项分布, prv_y指在等位基因dropout, 扩增不均衡等因素之后vaf与概率之间的关系
            self.prv_y = self.prv_y_b
        else:
            # 符合贝塔二项分布。
            self.prv_y = self.prv_y_bb

        # 细胞个数, 变异位点, 总深度, variant深度
        self.cells = list(df_total['cell_id'].values)
        self.muts = list(df_total.columns[1:])
        self.df_total = self.df_total.set_index(['cell_id'])
        self.df_alt = self.df_alt.set_index(['cell_id'])

        if (self.df_total.values - self.df_alt.values).min() < 0:
            raise Exception('total reads must be greater than or equal to alternate reads!')

        # Sigma: 发生dropout之前,单体和双体的vaf可能值,(0,1/2,1)是单体可能值, (0,1/4,1/2,3/4,1)是双体可能值
        self.Sigma = ((0, 1/2, 1), (0, 1/4, 1/2, 3/4, 1))
         # Theta: 发生dropout之后,单体和双体的vaf可能值,(0,1/2,1)是单体可能值, (0, 1/4, 1/3, 1/2, 2/3, 3/4, 1)是双体可能值
        self.Theta = ((0, 1/2, 1), (0, 1/4, 1/3, 1/2, 2/3, 3/4, 1))

        # P(x|z) z为单体和双体的标签, z=0单体, z=1双体。 x指发生dropout之前,各个位置单体和双体所能取得所有vaf的集合。 
        # px_z = 位置:(0,0),(1/2,0),(1,0):0 单体
        # px_z = 位置:(0,1),(1/4,1),(1/2,1),(3/4,1),(1,1):0 双体
        self.px_z = {x: 0 for z in [0,1] for x in itertools.product(self.muts, self.Sigma[z], [z])}

        # 计算假阳性, 假阴性和精度(precision)
        if self.alpha_fn == None or self.alpha_fp == None or self.precision == None:
            # get vaf numpy array
            vaf_values = (self.df_alt / self.df_total).values
            vaf_values = vaf_values[~np.isnan(vaf_values)]

            # filter vafs less than 0.15 and get alpha_fp
            if self.alpha_fp == None or self.precision == None:
                # 利用vaf<=0.15的点通过矩估计确定的贝塔分布的α,β值, mean1是这个贝塔分布的期望, prec1不清楚是什么值, 期望作为假阳性
                mean1, prec1 = getBetaMOM(vaf_values[vaf_values <= 0.15])
                self.alpha_fp = mean1
                # 利用vaf>=0.85的点通过矩估计确定的贝塔分布的α,β值, mean2是这个贝塔分布的期望, 期望作为假阴性
            if self.alpha_fn == None or self.precision == None:
                mean2, prec2 = getBetaMOM(vaf_values[vaf_values >= 0.85])
                self.alpha_fn = 1 - mean2
                # 利用vaf在0.15-0.85之间的点通过矩估计确定的贝塔分布的α,β值, mean3是这个贝塔分布的期望, 精度为三个精度的中位数
            if self.precision == None:
                mean3, prec3 = getBetaMOM(vaf_values[(vaf_values > 0.15) & (vaf_values < 0.85)])
                self.precision = np.median([prec1, prec2, prec3])

        print(f"alpha_fn = {self.alpha_fn}")
        print(f"alpha_fp = {self.alpha_fp}")
        print(f"precision = {self.precision}")

        if self.mu_hom == None or self.mu_hetero == None:
            estimate = True
        else:
            estimate = False

        # 每个位置遍历
        for m in self.muts:
            if estimate:
                # 总深度, 变异深度
                reads = self.df_alt[m]
                total = self.df_total[m]
                vaf = pd.DataFrame(reads/total)
                
                # 该位置所有细胞vaf确定纯合, 杂合, 突变纯合
                loh_cells = vaf.loc[vaf[m] > 0.85]
                wt_cells = vaf.loc[vaf[m] < 0.15]
                het_cells = vaf.loc[(vaf[m] >= 0.15 )& (vaf[m] <= 0.85)]

                
                # #count the total number of non-na VAF cells for variant m
                non_na_cells =  loh_cells.shape[0] + wt_cells.shape[0] + het_cells.shape[0]
                est_wt_rate = wt_cells.shape[0]/non_na_cells
                est_loh_rate = loh_cells.shape[0]/non_na_cells
                est_het_rate = het_cells.shape[0]/non_na_cells
       
                #将纯合, 杂合, 突变纯合率赋给上文px_z, px_z[m, 0,0]: 单体vaf=0; px_z[m,1/2,0]: 单体vaf=1/2; px_z[m,1,0]: 单体vaf=1
                # 之后的矩阵最后一位代表单体双体, [ , , 0]代表单体, [ , ,1]代表双体
                self.px_z[m, 0,0] = est_wt_rate
                self.px_z[m, 1/2, 0] =  est_het_rate
                self.px_z[m, 1, 0] = est_loh_rate


            else:
                self.px_z[m, 0,0] = 1 - self.mu_hom - self.mu_hetero
                self.px_z[m, 1/2,0] =  self.mu_hetero
                self.px_z[m, 1, 0] = self.mu_hom
            
        # 计算双体的P(x|z)即px_z, 根据两个单体的值合起来计算双体
        norm_const = {}
        for m in self.muts:
            # a,b为两个单体
            for a,b in itertools.product(self.Sigma[0], repeat = 2):
                c = (a + b)/2
                # 判断两个单体组成的vaf值是否在双体的vaf值中
                if c in self.Sigma[1]:
                    # 将两个单体的概率相乘，得到双体的概率(该vaf值下)
                    self.px_z[m,c,1] += self.px_z[m,a,0] * self.px_z[m,b,0]
            # norm_const用于归一化
            norm_const[m] = sum([self.px_z[m,a,0] * self.px_z[m,b,0] for a,b in itertools.product(self.Sigma[0], repeat = 2)])

        for m in self.muts:
            for c in self.Sigma[1]:
                # 归一化
                self.px_z[m,c,1] /= norm_const[m]
        
        # py_xz即P(y|x,z),y指的是发生dropout之后的vaf, x指的是发生dropout之前的vaf, z指单双体。
        # 指发生dropout之前,各个位置单体和双体所能取得所有vaf的集合,与px_z类似.
        self.py_xz = {x: 0 for z in [0,1] for x in itertools.product(self.Theta[z], self.Sigma[z], [z])}

        # beta是dropout率, 指所有情况下发生dropout后的概率。
        # 例如py_xz[0, 1/4,   1]指的是双体未发生drop之前vaf是1/4, 发生drop之后vaf是0的概率
        self.py_xz[0,   0,   0]= 1 - self.beta**2
        self.py_xz[0,   1/2, 0]= self.beta * (1 - self.beta)
        self.py_xz[1/2, 1/2, 0]= (1-self.beta)**2
        self.py_xz[1,   1/2, 0]= self.beta * (1 - self.beta)
        self.py_xz[1,   1,   0]= 1 - self.beta**2	
        self.py_xz[0,   0,   1]= 1 - self.beta**4
        self.py_xz[0, 1/4,   1]= self.beta * (1 - self.beta)**3 + 3 * self.beta**2 * (1 - self.beta)**2 + 3 * self.beta**3 * (1 - self.beta)
        self.py_xz[1/4, 1/4, 1]= (1 - self.beta)**4 
        self.py_xz[1/3, 1/4, 1]= 3*self.beta*(1 - self.beta)**3
        self.py_xz[1/2, 1/4, 1]= 3*self.beta**2 * (1 - self.beta)**2
        self.py_xz[1,   1/4, 1]= self.beta**3 * (1 - self.beta)
        self.py_xz[1/3, 1/2, 1]= 2 * self.beta * (1 - self.beta)**3
        self.py_xz[1/2, 1/2, 1]= (1-self.beta)**4 + 4 * self.beta**2 * (1 - self.beta)**2
        self.py_xz[2/3, 1/2, 1]= 2 * self.beta * (1 - self.beta)**3
        self.py_xz[1,   1/2, 1]= self.beta**2 * (1 - self.beta)**2 + 2 * self.beta**3 * (1 - self.beta)
        self.py_xz[0,   3/4, 1]= self.py_xz[1,   1/4, 1]
        self.py_xz[1/2, 3/4, 1]= self.py_xz[1/2, 1/4, 1]
        self.py_xz[2/3, 3/4, 1]= self.py_xz[1/3, 1/4, 1]
        self.py_xz[3/4, 3/4, 1]= self.py_xz[1/4, 1/4, 1]
        self.py_xz[1,   3/4, 1]= self.py_xz[0,   1/4, 1]
        self.py_xz[1, 1, 1] = 1 - self.beta**4



        self.doublet_result = None

        if self.delta == 0:
            self.threshold = sys.float_info.max
        elif self.delta == 1:
            self.threshold = -sys.float_info.max
        else:
            self.threshold = math.log((1 - self.delta) / self.delta)

    def solve(self):

        self.doublet_result = {}
        self.logprobs = {}
        for cell in self.cells:
            # 逐个细胞计算
            self.logprobs[cell, 0] = self.prv_z(cell, 0)
            self.logprobs[cell, 1] = self.prv_z(cell, 1)

            # 确定是否doublet
            if self.logprobs[cell, 1] - self.logprobs[cell, 0] > self.threshold:
                self.doublet_result[cell] =  'doublet'
            else:
                self.doublet_result[cell] = 'singlet'

    # 计算总的概率值, z=0和z=1分别计算
    def prv_z(self, cell, z):

        log_prob_sum = 0
        for mut in self.muts:
            v = self.df_alt.loc[cell, mut]
            r = self.df_total.loc[cell, mut] - v
            # 总深度和变异深度都是0时候的处理
            if self.missing and r + v == 0:
                if z == 0 and self.beta > 0:
                    log_prob_sum += 2 * math.log(self.beta)
                if z == 1 and self.beta > 0:
                    log_prob_sum += 4 * math.log(self.beta)
                continue
            prob_sum = 0

            # dropout之前的vaf取值 0, 1/2, 1(z=0)
            for x in self.Sigma[z]:
                prob_sum_x = 0

                # 找到发生dropout之后对应的vaf 0, 1/4, 1/2, 3/4, 1(z=0)
                for y in self.Theta[z]:
                    # 如果这种情况会出现，则用乘法公式计算似然
                    prob_sum_x += self.prv_y(r, v, y) * self.py_xz[y, x, z]
                prob_sum += self.px_z[mut, x, z] * prob_sum_x

            # 计算了所有可能出现的似然值的和, 计算过程主要是条件概率相乘
            log_prob_sum += math.log(prob_sum)
        return log_prob_sum

    # 二项分布(未使用)
    def prv_y_b(self, r, v, y):
        yprime = self.alpha_fp + (1 - self.alpha_fp - self.alpha_fn) * y
        return nCr(r+v, v) * (yprime ** v) * ((1-yprime) ** r)

    # 贝塔二项分布, 根据vaf读数计算概率, y的取值分别是0,1/2,1(单体), 0,1/4,1/2,3/4,1(双体)
    def prv_y_bb(self, r, v, y):
        # y修正值: p=y*(1−αfn)+(1−y)*αfp, y是发生drop之后的的vaf, 指该位点上有alt等位基因的概率(原文是Specifically, 
        #  the probability pi,j of producing a copy with the alternate allele at locus j)
        yprime = self.alpha_fp + (1 - self.alpha_fp - self.alpha_fn) * y
        if yprime == 0:
            yprime = 0.001
        if yprime == 1:
            yprime = 0.999
        # 不清楚为什么得到的α, β值
        alpha = self.precision*yprime 
        beta = self.precision - alpha
        n = r + v 
        #print(f"n:{n} r:{r} v:{v} p:{y} alpha:{alpha} beta:{beta}")

        # 分子
        num = math.lgamma(n+1) + math.lgamma(v+alpha) + math.lgamma(n-v+beta) + math.lgamma(alpha+beta)
        # 分母
        den = math.lgamma(v+1) + math.lgamma(n-v+1) + math.lgamma(n+ alpha + beta) + math.lgamma(alpha) + math.lgamma(beta)

        #这个概率计算的是, n总读数中有v个变异位点时的概率值, 先计算了组合数Cnv, 再乘以v发生的概率(贝塔分布)。具体是n,v符合二项分布，但v的概率又符合贝塔分布
        prob = math.exp(num- den)

        return prob

    def writeSolution(self, outputFile):

        with open(outputFile, 'w') as output:
            output.write("cell_id\tprob_z0\tprob_z1\tprediction\n")
            for cell in self.cells:
                output.write(f"{cell}\t{self.logprobs[cell, 0]}\t{self.logprobs[cell,1]}\t{self.doublet_result[cell]}\n")

    def likelihood(self):
        likelihood = len(self.cells) * math.log(1 - self.delta) 
        for cell in self.cells:
            if self.doublet_result[cell] == 'doublet':
                likelihood += self.logprobs[cell,1] - self.logprobs[cell,0] - self.threshold
        return likelihood

# 利用平均数，方差矩估计β分布参数, x_alpha/(x_alpha + x_beta)是这个贝塔分布的期望。第二个值不清楚
def getBetaMOM(x):
    
    m_x = np.mean(x)
    s_x = np.std(x)
    x_alpha = m_x*((m_x*(1 - m_x)/s_x**2) - 1)
    x_beta = (1 - m_x)*((m_x*(1 - m_x)/s_x**2) - 1) 
    
    return x_alpha/(x_alpha + x_beta), x_alpha + x_beta

# Cnr组合数, 从n个数中取得v个
def nCr(n,r):
    f = math.factorial
    #print(n, r)
    return f(n) // f(r) // f(n-r)

def main(args):

    df_total = pd.read_csv(args.inputTotal)
    df_alt = pd.read_csv(args.inputAlternate)

    if len(df_total) != len(df_alt):
        raise Exception("number of cells in the two input files do not match!")

    if len(df_total.columns) != len(df_alt.columns):
        raise Exception("number of cells in the two input files do not match!")

    ncells = len(df_total)
    npos = len(df_total.columns) - 1

    if args.verbose:
        print(f"number of cells is {ncells}")
        print(f"number of mutation positions is {npos}")

    solver = doubletFinder(df_total, df_alt, delta = args.delta, beta = args.beta, missing = args.missing, 
                           mu_hetero = args.mu_hetero, mu_hom = args.mu_hom, alpha_fp = args.alpha_fp, alpha_fn = args.alpha_fn,
                           verbose = args.verbose, binom = args.binom, precision = args.prec)

    solver.solve()

    solver.writeSolution(args.outputfile)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputTotal", type=str, help="csv file with a table of total read counts for each position in each cell")
    parser.add_argument("--inputAlternate", type=str, help="csv file with a table of alternate read counts for each position in each cell")
    parser.add_argument("--delta", type=float, default=0.1, help="expected doublet rate [0.1]")
    parser.add_argument("--beta", type=float, default=0.05, help="allelic dropout (ADO) rate [0.05]")
    parser.add_argument("--mu_hetero", type=float, help="heterozygous mutation rate [None]")
    parser.add_argument("--mu_hom", type=float, help="homozygous mutation rate [None]")
    parser.add_argument("--alpha_fp", type=float, help="copy false positive error rate [None]")
    parser.add_argument("--alpha_fn", type=float, help="copy false negative error rate [None]")
    parser.add_argument("-o", "--outputfile", type=str, help="output file name")
    parser.add_argument("--noverbose", dest="verbose", help="do not output statements from internal solvers [default is false]", action='store_false')
    parser.add_argument("--binomial", dest="binom", help="use binomial distribution for read count model [default is false]", action='store_true')
    parser.add_argument("--prec", type=float, help="precision for beta-binomial distribution [None]")
    parser.add_argument("--missing", dest="missing", help="use missing data in the model? [No]", action = 'store_true')
    parser.set_defaults(missing=False)
    parser.set_defaults(binom=False)
    parser.set_defaults(verbose=True)

    args = parser.parse_args(None if sys.argv[1:] else ['-h'])

    main(args)
