"""Compact vector overview inspired by the figures inside the reference papers.

Uses original shapes and labels; no screenshots or website illustrations.
Run from the project root with .venv/bin/python latex/scripts/build_concept_figure.py.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle

OUT=Path(__file__).resolve().parents[1]/'figures'
OUT.mkdir(parents=True,exist_ok=True)
INK='#263647'; MUTED='#566475'; BLUE='#e8f1fa'; GREEN='#e7f2e9'; GRAY='#f1f2f4'; SAND='#f8efdf'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(6.7,1.85))
ax.set(xlim=(0,6.7),ylim=(0,1.85));ax.axis('off')

def box(x,y,w,h,fill='white',edge='#bbc3cc',dash=False,r=.08):
 p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.025,rounding_size={r}',
   facecolor=fill,edgecolor=edge,linewidth=.75,linestyle=(0,(3,2.5)) if dash else '-')
 ax.add_patch(p);return p

def label(x,y,t,size=8,weight='normal',color=INK,ha='center',va='center'):
 return ax.text(x,y,t,ha=ha,va=va,fontsize=size,fontweight=weight,color=color,linespacing=1.35)

def arrow(a,b,style='-|>',dash=False,color=INK,lw=1.1):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=10,linewidth=lw,
   linestyle=(0,(3,2.5)) if dash else '-',color=color,connectionstyle='arc3'))

# Readers and articles: simple original vector icons.
box(.04,.36,1.17,1.38,edge='#c7cdd4',dash=True,r=.12)
label(.625,1.56,'MIND inputs',8.5,'bold')
ax.add_patch(Circle((.33,1.21),.072,facecolor=INK,edgecolor='none'))
ax.add_patch(FancyBboxPatch((.21,.94),.24,.17,boxstyle='round,pad=0,rounding_size=.055',facecolor=INK,edgecolor='none'))
label(.73,1.18,'Reader\nhistory',7.5)
for x,y in [(.23,.55),(.28,.60)]:
 box(x,y,.34,.25,fill='white',edge=MUTED,r=.015)
 for yy,ww in [(y+.18,.23),(y+.13,.23),(y+.08,.14)]:ax.plot([x+.05,x+.05+ww],[yy,yy],lw=.55,color=MUTED)
label(.89,.70,'News\ncandidates',6.9)
arrow((1.24,1.05),(1.48,1.05))

# The four research questions share one action interface.
box(1.51,.36,2.37,1.38,edge='#b8c1cc',dash=True,r=.12)
label(2.695,1.56,'Choose an article',8.5,'bold')
for x,y,w,h,fill,title,desc in [
 (1.64,.97,1.01,.40,BLUE,'Q1  Allocation','A/B vs. bandits'),
 (2.76,.97,.99,.40,GREEN,'Q2  Content','Transfer to\nnew articles'),
 (1.64,.48,1.01,.40,GRAY,'Q3  SAE features','Explain policy\nchoices'),
 (2.76,.48,.99,.40,SAND,'Q4  LLM agent','With / without\na tool')]:
 box(x,y,w,h,fill=fill,edge='#cbd1d8',r=.06)
 label(x+w/2,y+.29,title,7.2,'bold');label(x+w/2,y+.105,desc,6.7)
arrow((3.92,1.05),(4.20,1.05))

# Learned environment: hidden probabilities generate observable feedback.
box(4.24,.36,1.14,1.38,fill=BLUE,edge='#a5b7ca',dash=True,r=.12)
label(4.81,1.56,'Click simulator',8.1,'bold')
# Minimal model symbol made from connected nodes.
for x,ys in [(4.60,[1.13,1.31]),(4.82,[1.06,1.22,1.38]),(5.04,[1.13,1.31])]:
 for y in ys:
  ax.add_patch(Circle((x,y),.026,facecolor='#5a7e9f',edgecolor='none',zorder=3))
for ya in [1.13,1.31]:
 for yb in [1.06,1.22,1.38]:ax.plot([4.60,4.82],[ya,yb],lw=.45,color='#a5b7ca',zorder=2)
for ya in [1.06,1.22,1.38]:
 for yb in [1.13,1.31]:ax.plot([4.82,5.04],[ya,yb],lw=.45,color='#a5b7ca',zorder=2)
label(4.81,.86,'Estimated response\nSample click / no click',6.9)
label(4.81,.53,'Check AUC,\ncalibration and gaps',6.8,color=MUTED)
arrow((5.42,1.05),(5.68,1.05))

box(5.72,.36,.93,1.38,edge='#bcc5cd',dash=True,r=.12)
label(6.185,1.56,'Evaluate',8.5,'bold')
for yy,t,c in [(1.25,'Clicks',BLUE),(.96,'Oracle regret',SAND),(.67,'95% intervals',GREEN)]:
 box(5.82,yy-.10,.73,.20,fill=c,edge='none',r=.035)
 label(6.185,yy,t,7.0)
# Feedback loop is observable reward only, never hidden true probabilities.
ax.plot([4.81,4.81,2.69],[.32,.13,.13],lw=.9,color=MUTED,ls=(0,(3,2.5)))
arrow((2.69,.13),(2.69,.34),dash=True,color=MUTED,lw=.9)
label(3.80,.13,'  observed feedback  ',6.8,color=MUTED).set_bbox(dict(facecolor='white',edgecolor='none',pad=.3))
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
for ext in ['pdf','svg','png']:
 fig.savefig(OUT/f'project_overview.{ext}',dpi=240,bbox_inches='tight',pad_inches=.015)
plt.close(fig)
