import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kruskal, mannwhitneyu, friedmanchisquare, wilcoxon, levene
import numpy as np
import os


data = pd.read_csv('statsSQ1.csv', sep=';')
# print(data)
# data = {
# "thingyone" : [5, 5, 4, 3, 5, 8, 8, 6, 7, 3],
# "thingytwo" : [3, 4, 5, 6, 7, 8, 9, 2, 4, 4]}
# #
df = pd.DataFrame(data)
print(df)
#
# thing1 = df.thingyone
# thing2 = df.thingytwo
#
# def plot(data, title):
#     sns.set(font_scale=2)
#     sns.set_style(style='white')
#
#     ax = sns.violinplot(data=data, cut = 0, bw = 0.35, palette = 'hls').set(title=title)
#     # sns.color_palette("pastel")
#     plt.ylabel("Average runtime")
#     plt.show()
#     # sns.boxplot(data=data, x="heuristic", y=y_axis, showfliers = False).set(title=title)
#
#     # plt.show()
#
# plot(df, title = 'even more idk')
#
#

#
# """From here statistics"""
# print(f"shapiro-test: {shapiro(thing1)}")
# print(f"shapiro-test: {shapiro(thing2)}")
# print('Shapiro p-value greater than 0.05 --> normally distributed')
# print(f"levenes test: {levene(thing1, thing2)}")
# print('Levene p-value greater than 0.05 --> equal variances')
