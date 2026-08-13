# CELL ATLAS — 细胞图谱
每类细胞:它是什么(生物对应)、接谁连谁、对机器本身的作用(拿掉会怎样)、实现思路。
Every cell type: what it is (biological counterpart), what it wires to, what it does *for the
machine itself* (what breaks without it), and how the code realizes it in plain language.
引擎 Engine: `neo_core.py` · 协议层 Protocol rigs: `value_confirm2.py` 等 · 许可 License: Apache-2.0

---

## 0. 一条总规则 / The one rule

所有突触只服从一条三因子规则 (All synapses obey one three-factor rule):

    Δw = η · U · e · M
    e  = 资格迹 eligibility(突触前后同时活跃的近期痕迹,按区域均值中心化,τ≈3 拍)
    U  = 1 + Q_NA + Q_ACh(唤醒增益 arousal gain)
    M  = Q_DA − Q_5HT(带符号裁决 signed verdict)

Q 是受体占据(Michaelis–Menten 饱和),**只由真实放电释放的递质填充**——一个不放电的细胞
什么也释放不了,所以任何通路都不可能从"从未物理到达的信号"里学习。这一条防作弊性质
不是规则,是解剖。
Receptor occupancies are filled only by spike-contingent release: a silent cell releases
nothing, so no pathway can learn from a signal that never physically arrived. This
anti-cheating property is anatomy, not a rule.

---

## 1. 感觉细胞 sens / Sensory cells
- **生物对应**:感觉皮层输入层。Sensory cortex input layer.
- **接线**:世界线索 → sens → 记忆细胞(mem)。Cue current in; projects to memory.
- **作用**:把世界的符号变成组织内的活动模式。没有它,世界对大脑不存在。
- **实现思路**:每个线索符号点亮一小群 sens 细胞;`encode_seq()` 把线索按拍送入。

## 2. 记忆细胞 mem / Memory (plateau) cells
- **生物对应**:海马 CA1 平台电位细胞(BTSP, Bittner 2017)。
- **接线**:sens → mem(可塑);mem → 回答细胞团(可塑,价值知识的主要居所)。
- **作用**:把刚发生的事保持成一个"平台态"——不是缓冲区,是动力学本身的驻波。
  剪掉:上下文消失,每一拍都是失忆的第一拍。
- **实现思路**:越过阈值的细胞进入自维持平台(`plat`),随时间缓慢衰减;
  试次之间由协议层复位(短时程),长时程权重永不复位。

## 3. 回答细胞团 outs / Answer cohorts (the action race)
- **生物对应**:运动/前运动皮层竞争群;侧抑制赛跑 = 决策的物理形态。
- **接线**:mem → outs(可塑,这里住着"该做什么");outs 团间互抑;胜者越过发放阈值=动作。
- **作用**:**决策不是计算出来的,是赛出来的**。8 个团 = 8 个可能动作(觅食世界里
  1=富矿 3=贫矿 5=空矿 6=毒矿,其余=游荡)。剪掉:有想法,无行动。
- **实现思路**:每拍向所有团注入 OU 噪声+突触驱动,先过 `THETA_EMIT` 者当选;
  当选团的适应电流 `adC` 上升(说过的话暂时不想再说)。

## 4. 多巴胺比较器 pcmp → DA / Dopamine comparator
- **生物对应**:VTA 多巴胺细胞;肠道迷走标记线(gut→NTS→VTA)。
- **接线**:肠道线(进食时的 dE)→ pcmp;pcmp 放电 → 全组织 DA(Q_DA)。
- **作用**:好消息的唯一入口。**按需要计价**:dE = min(meal, 1−E),同一顿饭,
  饿时值钱、饱时几乎免费——"价值相对于身体状态"由计价单位免费实现。
  剪掉(假身体臂):终生随机水平,什么也学不会(C1 对照,10/10)。
- **实现思路**:协议层把身体真实吸收的能量差乘 k_body=3.2 作为教学电导注入 pcmp,
  只在资格迹存活的教学窗(TEACH 拍)内。

## 5. 血清素比较器 ncmp → 5HT / Serotonin comparator
- **生物对应**:中缝核 5HT 细胞;脊髓-臂旁核痛觉标记线。
- **接线**:两根身体线汇入——痛觉线(伤害)与饥痛线(匮乏);ncmp 放电 → 全组织 5HT。
- **作用**:坏消息的唯一入口,价值数轴的负半轴。**按伤害计价**(与身家无关——
  若按账本计价,赤贫者恰在毒最要命时被麻醉,实测出死亡螺旋后修正)。
  **同报原则**:甜与即效毒必须在同一教学窗同报,否则短资格迹给甜头结构性优势
  (化学内建时间折扣)。剪掉:追逐糖衣毒(C4 断线对照,~220 口 vs ~14 口)。
- **实现思路**:毒事件注入 k_harm×毒强度;重饿(hunM>0.5)时向"正在执行的策略"
  注入 k_pang×hunM——挨饿的每一刻都在给现行路线记过(AgRP 负效价,Betley 2015)。

## 6. NA / ACh 池 — 唤醒与重要性 / Arousal pools
- **生物对应**:蓝斑(NA)、基底前脑(ACh)。
- **接线**:比较器/感觉侧 → NA/ACh 细胞 → 全组织 U 因子。
- **作用**:不改方向,改**幅度**——"此刻重要,学快点"。
- **实现思路**:占据进 U = 1 + Q_NA + Q_ACh,乘在学习率上。

## 7. 饥饿细胞 hun / Hunger cells
- **生物对应**:下丘脑 AgRP 神经元。
- **接线**:身体能量缺口 → hun(**先天调谐,学习不可改写**——"什么是饿"不容谈判);
  hun 放电释放肽通道 H(第五调质)到回答细胞团受体。
- **作用三重**:(a) **饱足刹车**:赛跑驱动乘 g = h0+(1−h0)·clip(hGain·Q_H)——
  饱了整台赛跑安静下来(十条命约一万个饱态相遇 0 参与;必需,因为习惯不理会教学消退);
  (b) **绝望探索**:赛跑噪声乘 (1+2·hunM)——饿急了才乱撞(压力下改写的"摇");
  (c) 饥痛线的信号源(见 §5,"凿")。
- **实现思路**:六个细胞的阈值梯度铺在缺口轴上(`hunTheta`),
  activity = tanh(4·(deficit−θ));肽以率标度释放(占据≈饥饿的模拟量)。

## 8. 归一化池 nrm 与新奇通路(默认关)/ Normalization & novelty (off by default)
- **生物对应**:PV+ 前馈抑制的除法归一化(Carandini-Heeger);好奇 = 论文二战线。
- **本论文全部结果不依赖这两条**;开关默认关,记录在案。

---

## 9. 三根导线一个刹车——价值机的全部 / The whole value machine
基因组只付了这些钱 (the genome pays for exactly this much):

| 导线 Wire | 计价 Pricing | 去向 Target | 剪断的死法 What severing does |
|---|---|---|---|
| 肠道→DA | 按需要 need (dE=min(meal,1−E)) | pcmp | 什么也学不会 (chance forever) |
| 痛觉→5HT | 按伤害 damage(无视身家) | ncmp | 追逐糖衣毒 (chases sweet poison) |
| 饥痛→5HT | 按匮乏 deficit(记过现行策略) | ncmp | 安于不解饿的习惯 (locks on inadequate options) |
| 肽刹车 | — | 赛跑增益 race gain | 饱汉无度 (never rests) |

导线里没有智能,只有化学;智能是后果在突触上雕出来的。
The wires contain no intelligence, only chemistry; intelligence is what consequences carve.

---

## 10. 常数出处 / Constant provenance
没有一个常数是调考分调出来的;每个都是隔离台上的单次测量,寒冬协议在全部常数冻结下运行。
- k_body = 3.2:典型一餐 0.25 ↦ 隔离台验证过的 0.8 教学幅度类。
- k_harm = 3.2:与 k_body 对称(同一幅度类)。
- k_pang = 0.6:置重饿(h→1)的教学幅度于贫餐(0.26)与富餐(0.8)幅度之间——这是相对世界报酬结构的一次性选择,论文如实申明;结论对它的敏感性由它所处的两个幅度界定。
- h0 = 0.15 / hGain = 2.46:实测二级信使增益(占据→增益的换算)。
- KDESP = 2.0:全饿时探索噪声三倍(预注册单常数)。
- τ_metab = 500、餐 0.25/0.08/0.03、毒 0.05−0.25:世界物理,一处定义。

---

## 11. 物理在场但禁用的机构 / Present-but-disabled machinery
公开引擎中还建有以下机构,本文全部结果都在它们关闭态下取得;逐一披露(旗标=默认值):
The released engine also contains the following, all OFF for every result in this paper:
- 张力起搏细胞 tonic pacemakers(tonicMod=0):调质核团内禀基线放电;自发行为战线,后续论文。
- 缰核样凹陷线 dip wire(dipWire=0):把"预期落空"变成 DA 基线下的暂停(负 RPE);未验证。
- 对立价值细胞器官 value-cell organ(valcells=0):vapp/vave 双侧账本,48 细胞;本文价值机不用账本。
- 意志门 volition gate(volition=0):旧架构的行动闸门;本文由赛跑+肽门取代。
- 新奇/归一化通道 novelty & normalization(curioSyn=0, nrm 随之):好奇心战线,后续论文。
这些机构对本文任何数字无贡献;它们的在场是工程史实,如实列出。
