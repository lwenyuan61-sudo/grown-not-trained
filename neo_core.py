"""Wanning-neo core engine v0.1 — faithful Python port of the validated rule set (SPEC.md).
Pure local rules, no backprop. Vectorized numpy; scale is a config parameter (512 -> 16k -> 370k).
Provenance: seq_engine.js (validated stages 0/A/B2/C2/D) + self_initiated.js (volition gate, reserved).
"""
import numpy as np

P = dict(
    gain=0.135, Khalf=0.15, tauX=3.0, tauE=3.0, eta=0.04,
    # plateau memory (SPEC 1.3)
    gD=1.0, theta0=0.15, thetaRel=1.35, scaleD=0.45, kVetoI=4.0, tauP=200.0, kP=0.40,
    gP=1.35, kVeto=0.12, kAHP=0.0005, tauAH=25.0, gA=0.008,
    etaDendLTP=0.004, etaDendLTD=0.002, wDendMax=0.62,
    tauHome=3000.0, homeTarget=0.08, kHome=1.0, kEnc=0.3, tauBuf=5.0,
    # position system (SPEC 1.4)
    sigmaPos=0.08, sigmaPosC=0.05, posAmp=0.8, exprGate=1,
    # race output (SPEC 1.5)
    tauAcc=5.0, kGo=0.36, LEAK=0.25, THETA_EMIT=0.45, gAd=0.6, tauAd=40.0, MAXW=30,
    sigmaOut=0.05, sigmaCoh=0.30, tauOU=4.0,
    # reward / D1-D2 (SPEC 1.6-1.7)
    TEACH=5, rWrong=-0.25, rOver=-1.0, etaV=0.05, postCross=3, slotGap=10,
    lamD1=0.005, wCapRace=1.1, D1budget=8.0, etaD2fac=0.5, lamD2=0.006, D2budget=3.0,
    kNegSlope=2.5,
    CUE=5, GAP=4, ITI=6,
    # structural turnover: overcomplete birth + activity-dependent pruning + rewiring (SPEC 1.12).
    # Biology: cortex overproduces synapses ~2x then prunes ~40-50% activity-dependently; adult
    # spine turnover continues. Local rule: a synapse that is weak AND unused gets its slot rewired
    # to a fresh random presynaptic cell in the same source region (synapse count conserved).
    # turnover=0 -> exact prior behavior (safety gate).
    turnover=1, tauUse=800.0, wPrune=0.04, uPrune=0.003, pruneFrac=0.02, warmupTrials=100,
    # useGate: visit-rate-INVARIANT usage accounting. 0 = absolute-tick leaky average (original;
    # implicitly assumes single-lesson visit rates: a rare-visit state's edges decay below uPrune
    # between visits and get churned — measured multi-lesson poison). 1 = the trace only advances
    # in proportion to local activity g=clip(act_pre+act_post,0,1): silent neighborhood = frozen
    # account (no evidence, no verdict); active neighborhood = normal accounting. Biology: synapse
    # elimination is activity/Ca-dependent (binocular deprivation prunes nothing; monocular
    # deprivation prunes the silent eye BECAUSE cortex is active — post-activity drives it);
    # disused-but-uncontested synapses are stable (memories survive disuse).
    useGate=1,
    # turnD2: 0 = D2 frozen (excluded from turnover); 1 = same rule as excitatory (WRONG for
    # inhibition: successful suppression erases its own use signature — measured, seed42 + overruns);
    # 2 = INVERTED usage for inhibitory edges: use = act_pre*(1-act_post)  (pre firing while post
    # stays silent = the synapse is doing its job; Lee's inversion).
    turnD2=1,
    # silentBirth: rewired (nascent) synapses are born near-silent — biology: new spines are
    # AMPA-silent (NMDA coincidence-detector only, no current) until LTP unsilences them.
    # Eligibility is activity-based (cen), not weight-based, so a silent newborn can still be
    # captured by reward and grown. Kills the "nascent lawn" that drowns the D2 stopping wall.
    silentBirth=1, nbLo=0.002, nbHi=0.012,   # nascent-synapse birth conductance range
    sigFloor=0.05,                            # exploration floor: a punished state may go quiet, never dead
    # LAYER 4 — selective consolidation (late-LTP elastic anchor, Fable spec §4). Diagnosed need:
    # learned D1 pathways oscillate build/teardown (decay outpaces earning during silence streaks);
    # global slow decay fixes it but bluntly (side effect: value/stop equilibrium shifts). The
    # biological answer: synapses repeatedly CAPTURED by salient reward get a structural anchor
    # (spring back), everything else keeps fast decay. anchor=0 -> exact prior behavior.
    anchor=1, tauPRP=400.0, kPRP=0.5, thetaPRP=0.9, scalePRP=0.8, kDrain=2.0,
    kCap=0.08, kWa=0.30, tauProt=60000.0, kAnc=0.06, protSafe=0.35,
    # RECONSOLIDATION: capture is CONSISTENCY-gated. Growth only on reward-coincident eligibility
    # (M>0); an edge eligible during a PUNISHED action destabilizes (melts) in proportion to
    # |elig|·|M-| — no protein needed to destabilize (reactivation+prediction-error suffices;
    # restabilization needs protein). Nader/Sara reconsolidation. Consequence: single-lesson
    # correct paths (only ever in rewarded events) consolidate; multi-lesson monopolies (eligible
    # in punished emissions) melt; early exploration (errors frequent) can't consolidate at all.
    kMelt=0.12,
    # CREB GATE (consolidation v3, biology-faithful): capture is gated by a PER-CELL SLOW nuclear
    # integrator (CREB analog) driven by receptor occupancy — a_i·(Q_DA+Q_NA) kinase side (PKA)
    # minus a_i·Q_5HT phosphatase side (calcineurin) — with a time constant of ~100+ trials.
    # The SLOWNESS is the consistency filter: a monopoly punished half the time never sustains
    # CREB; and by the time CREB crosses threshold the weights have CONVERGED, so the anchor
    # snaps to the converged state (wa := w) — no stale-birth-anchor drag. Kinase/phosphatase
    # tug-of-war + melt (reconsolidation) stay. crebGate=0 -> prior (RECON) behavior.
    crebGate=1, tauCREB=20000.0, kCreb=25.0, kPhos=1.5, thetaCREB=0.30,
    # value-side consolidation: aversion/appetite learned from real outcomes is FAST; relaxation of
    # value during silence (no outcome = weak evidence) is SLOW (extinction slower than acquisition).
    etaVsil=0.1,
    extinctAsym=1,   # 0 = original flat critic_learn(0) on silence (pre-AB6 behavior)
    # VOLITION GATE (SPEC 1.9, validated in self_initiated.js): a GO population of
    # integrate-to-threshold cells decides WHETHER she engages at all. Confidence drive from the
    # answer populations + OU babbling noise accumulates against an intrinsic leak (rheobase) set
    # between untrained (~0.40) and trained (~1.24) drive: don't know -> never crosses -> she
    # declines (silence); know -> crosses -> she answers. Paused = no ticks = mute. Basal-ganglia
    # go-gate + Schurger stochastic accumulator. volition=0 -> prior behavior (forced engagement).
    volition=0, kGoV=0.7, LEAKV=0.42, THETA_GOV=0.32, sigmaGoV=0.40, tauGoV=4.0, tauAccV=5.0, VMAXW=40,
    # VALUE CELLS (ventral-striatum organ; critic biologization, SPEC 1.7 debt). Two OPPONENT
    # subpopulations in ONE region (region-mean centring needs their difference for variance):
    # vAPP learns on reward side (dw ∝ elig·M+, D1-family), vAVE on punishment side (dw ∝ elig·M−,
    # D2-family) — value corrections come from the OPPONENT growing, each side stays monotone+decay.
    # v = kVal·(mean act vAPP − mean act vAVE). valcells=1 builds the organ in SHADOW (host critic
    # still drives behavior; VAL learns alongside — V1 accuracy probe). valTD=1 wires VAL→comparators
    # (r − v computed by the brain itself; host crutches deleted) — only after V1 passes.
    valcells=0, valTD=0, valLam=0.0001, kVal=4.5, valEta=0.15, valEvid=0,  # valEvid: First-Law evidence gate on opponent growth (A/B under test)
    extinctEvid=0, extinctRef=0.5,  # counterconditioning: fear extinction rate scales with positive evidence (A/B under test)
    allies=0, h0Want=0.15, kBrave=0.0, hunGovW=1.0, vapGovW=1.0, kDespV=0.0, valSlowE=0, tauEV=25.0, usVave=0, valMemOnly=0, vavGov=0, kCert=1.0, kWallC=0.0, alwaysOn=0, lamScale=1.0, datPump=0, tauDAT=1.0, hasselmo=0, thetaACh=0.1, rulerFreeze=1, vScale=0, vSet=0.008, tauVH=1000.0, kVScale=3e-3, matAve=0.0, naWire=0, naW=0.737, criticOff=0, achWire=0, achW=0.700, rDirect=0, wallCert=0, cmpAdapt=0, kCmpA=1.5, tauCmpA=2000.0, fearAnchor=0, reliefCap=0, kRelCap=0.05, reliefCell=0, kRelCell=0.02, tauFearTr=10.0, hchan=0, hW=0.004, hGain=2.46, hRace=0, tdCross=1, curio=0, kCurio=0.5, habUp=0.5, tauHab=60000.0, curioSyn=0, curioPc=0.30, curioW=0.75, nrmInW=0.4, nrmOutW=1.0, kNrm=2.0, novOutW=1.22,  # measured: relay chain delivers 0.357 gov-equiv for novel cue (2026-08-11)  # curioW: end-to-end gain match to novB reference (measured 2026-08-11: wires delivered 1/12.4 of novB-equivalent drive at 0.06)  # alliesthesia + gate-ear calibration (knee: app 0.067 must cross, 0.03 must not — measured seed7/seed3 @ valLam 0.001): satiety gates value->volition multiplicatively (A/B under test)  # TEACHER'S PATIENCE: the critic must converge slower than policy learning + CREB induction (~300 trials), or surprise dies before maps/walls are consolidated (measured V2a-c cascade)  # valLam: 0.003 starved the organ (weights 0.49->0.03 in 100 trials; up-state death spiral, measured) — value memories must outlive their earning window
    # SPN-style value cells (biology rebuild): silent at rest, hard UP-STATE threshold (only a
    # constellation MATCH fires them), SPARSE-STRONG afferent sampling (K pres each, few-and-large
    # like mossy fibres/DG — 'selectivity beats coverage'), lateral inhibition keeps the code
    # sparse. Selectivity exists FROM BIRTH -> centred three-factor eligibility has real variance
    # (no chicken-and-egg). gVal/thetaUp computed from match arithmetic, not tuned.
    gVal=3.0, thetaUp=0.30, valK=8,   # placed at measured p87 of drive; winners loud (measured, 2026-08-08)
    # NEEDS SYSTEM step 1 (SPEC 2.x): body energy + interoceptive hunger cells + labeled-line
    # primary reward. The body is physiology, not cognition: hunger-cell tuning is INNATE
    # (hypothalamic AgRP analog — the definition of hunger may never be rewritten by learning),
    # and the ONLY positive-reward entry is the energy delta of real eating (gut->NTS labeled
    # line). Drive current wiring (hunger->volition/exploration) is step 2. needs=0 -> inert.
    # TONIC PACEMAKERS (Lee directive 2026-08-10: if biology has it, we build it — the substrate
    # for self-initiated action and thought). Biology: DA neurons pacemake ~4-5Hz tonic (dips
    # below baseline encode negative RPE); locus coeruleus, raphe, basal forebrain all tonic;
    # cortex is never silent. tonicMod: intrinsic drive + intrinsic noise on the four modulator
    # nuclei. dipWire: ncmp INHIBITS da (LHb->RMTg->DA anatomy) so omission = dopamine dip below
    # its own baseline — negative surprise without a serotonin event. sigBg: spontaneous cortical
    # tone on exc+mem (rest is alive; replay/imagination seed; Schurger fluctuations become
    # tissue-owned, not protocol-injected). All default OFF: goldens bit-safe.
    uncExplore=0, kUncUp=0.6, kSettle=0.015, tauStale=30000.0,  # uncertainty-directed exploration (Lee arc 2026-08-10): explore what is POORLY UNDERSTOOD, not what is merely unvisited
    tonicMod=0, tonicDA=0.06, tonicNA=0.036, tonicACH=0.042, tonicHT=0.048, sigTonic=0.05,  # calibrated: dip -0.21/burst +0.17 bidirectional range, rest Mr~0 (smoke 2026-08-10)
    dipWire=0, sigBg=0.0,
    needs=0, Esetpoint=0.7, tauMetab=3000.0, costEmit=0.004, mealSize=0.25, kMeal=4.0, kHunSig=0.4,
)
SLOTC = np.array([0.1, 0.3, 0.5, 0.7, 0.9])


class Brain:
    def __init__(self, seed, nSym=8, cps=5, nMem=160, nk=6, nPos=30, nExc=40, overcomplete=1.6):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.nSym, self.cps, self.nk = nSym, cps, nk
        blocks = {}
        p = 0
        def block(n):
            nonlocal p
            a = np.arange(p, p + n); p += n; return a
        blocks['sens'] = block(nSym * cps); blocks['exc'] = block(nExc); blocks['finh'] = block(max(4, nExc // 6))
        blocks['mem'] = block(nMem); blocks['minh'] = block(8); blocks['clr'] = block(4)
        blocks['posC'] = block(nPos)
        blocks['cohorts'] = [block(nk) for _ in range(nSym)]
        blocks['gov'] = block(8)                                             # volition GO population
        blocks['nrm'] = block(6 if P['curioSyn'] else 0)                    # PV-style normalization pool (cellization S1)
        blocks['nov'] = block(6 if P['curioSyn'] else 0)                    # novelty RELAY nucleus (Lisman-Grace; knife 3)
        blocks['vapp'] = block(24); blocks['vave'] = block(24)               # opponent value cells (SPN)
        blocks['hun'] = block(6)                                             # interoceptive hunger cells (innate)
        blocks['pcmp'] = block(4); blocks['ncmp'] = block(4); blocks['ucmp'] = block(4)  # visceral US comparator (needs era)
        for nm in ('da', 'na', 'ach', 'fht'): blocks[nm] = block(8)
        self.blocks = blocks; self.N = p
        self.outs = np.concatenate(blocks['cohorts'])
        # thresholds
        self.thr = np.full(self.N, 0.12)
        for nm in ('da', 'na', 'ach', 'fht'):
            c = blocks[nm]; self.thr[c] = 0.02 + 0.18 * np.arange(len(c)) / (len(c) - 1)
        # regions
        self.region = np.zeros(self.N, np.int32); r = 0
        for cells in (blocks['sens'], blocks['exc'], blocks['finh'], blocks['mem'], blocks['minh'],
                      blocks['clr'], blocks['posC'], self.outs, blocks['gov'], blocks['nrm'], blocks['nov'],
                      np.concatenate([blocks['vapp'], blocks['vave']]),      # ONE region: opponent difference = variance
                      blocks['hun'], blocks['pcmp'], blocks['ncmp'],
                      blocks['da'], blocks['na'], blocks['ach'], blocks['fht']):
            self.region[cells] = r; r += 1
        self.nreg = r
        # ---- connectivity (SPEC tables) ----
        src, dst, w, modc, dend, veto = [], [], [], [], [], []
        def conn(pre, post, pc, lo, hi, sign=1, mod=-1, dendf=False, vetof=False):
            pre = np.asarray(pre); post = np.asarray(post)
            mask = rng.random((len(pre), len(post))) < pc
            ii, jj = np.nonzero(mask)
            keep = pre[ii] != post[jj]
            ii, jj = ii[keep], jj[keep]
            src.extend(pre[ii]); dst.extend(post[jj])
            w.extend(sign * rng.uniform(lo, hi, len(ii)))
            modc.extend([mod] * len(ii)); dend.extend([dendf] * len(ii)); veto.extend([vetof] * len(ii))
        B = blocks
        conn(B['sens'], B['exc'], 0.15, 0.05, 0.25); conn(B['exc'], B['exc'], 0.08, 0.02, 0.15)
        conn(B['finh'], B['exc'], 0.20, 0.05, 0.35, -1); conn(B['exc'], B['finh'], 0.15, 0.03, 0.20)
        conn(B['sens'], B['mem'], 0.30, 0.20, 0.50, 1, -1, True, False)   # afferent -> dendrite
        conn(B['mem'], B['mem'], 0.05, 0.02, 0.10)
        conn(B['mem'], B['minh'], 0.50, 0.10, 0.30); conn(B['minh'], B['mem'], 0.60, 0.10, 0.40, -1, -1, False, True)
        conn(B['clr'], B['mem'], 0.90, 0.30, 0.60, -1, -1, False, True)
        oc = overcomplete   # overcomplete birth on the readout classes (turnover then sculpts them)
        mos = len(w); conn(B['mem'], self.outs, min(0.95, 0.40 * oc), 0.02, 0.10); moe = len(w)  # content -> symbol
        d1s = len(w); conn(B['posC'], self.outs, min(0.95, 0.60 * oc), 0.02, 0.10); d1e = len(w)          # D1 direct
        d2s = len(w); conn(B['posC'], self.outs, min(0.95, 0.60 * oc), 0.01, 0.05, -1); d2e = len(w)      # D2 indirect
        conn(B['exc'], self.outs, 0.20, 0.02, 0.10)
        # value-cell afferents: state (mem + posC) -> opponent value populations. PLASTIC, learned
        # by the D1/D2-family opponent rules (marked below by index range).
        statepool = B['mem'] if P['valMemOnly'] else np.concatenate([B['mem'], B['posC']])
        # valMemOnly: ventral striatum samples CONTENT afferents only (hippocampal/BLA 'what').
        # posC cells are cue-invariant and fire every trial -> their weights accumulate pure
        # CONTEXT value ('being at a patch = good'), a floor every option inherits free
        # (measured: EMPTY pinned at +0.17-0.23 = med's level in comfortable seeds, C5 unpassable).
        # Position-dependent value survives: mem cells are position-BOUND (memPos joint code).
        def spn_sample(post_cells):
            for j in post_cells:
                pres = rng.choice(statepool, P['valK'], replace=False)
                for i in pres:
                    src.append(int(i)); dst.append(int(j))
                    w.append(rng.uniform(0.35, 0.65))
                    modc.append(-1); dend.append(False); veto.append(False)
        vps = len(w); spn_sample(B['vapp']); vpe = len(w)
        vns = len(w); spn_sample(B['vave']); vne = len(w)
        rcs = len(w)
        # volition gate wiring: INNATE (non-plastic, inside the race-edge range).
        # outs->gov REMOVED (2026-08-09): the old 'confidence from answer populations' ear was
        # measured dead (mean activity carries no confidence) AND its content echo alone crossed
        # the rheobase — jamming the gate permanently open (V4 first run: engage 1.00 everywhere,
        # satiety ceiling flattened all meal deltas, VAL learned noise). The gate's ears are now
        # exactly Berridge's wanting: vAPP (value) + hun (drive). Nothing else.
        conn(B['gov'], B['gov'], 0.45, 0.05, 0.16)
        valall = np.concatenate([B['vapp'], B['vave']])
        conn(valall, valall, 0.30, 0.10, 0.25, -1)   # SPN lateral inhibition (innate): sparse code
        conn(B['hun'], B['gov'], 0.8, 0.08 * P['hunGovW'], 0.12 * P['hunGovW'])  # drive current: hunger pressures engagement (innate); sized SUB-rheobase when hunGovW<1 — desperate exploration must be stochastic (noise floor), not deterministic all-engage (measured: EMPTY@hungry 1.00)
        conn(B['vapp'], B['gov'], 0.8, 0.15 * P['vapGovW'], 0.25 * P['vapGovW'])   # incentive salience: VALUE is the volition gate's
        if P['vavGov']:
            # INDIRECT PATHWAY at the limbic gate (NAcc D2 -> ventral pallidum): approach =
            # appetite MINUS fear. Without this ear the gate is deaf to learned aversion —
            # (measured exam21: EMPTY net value -0.06/-0.10, fear fully learned, gate still
            # opens 0.58-0.67 because it hears vAPP alone).
            conn(B['vave'], B['gov'], 0.8, 0.15 * P['vapGovW'], 0.25 * P['vapGovW'], -1)
        # confidence substrate (the missing piece from the volition arc) — wanting = value + drive,
        # neither alone crosses the rheobase, together they do (additive AND-gating of Berridge's
        # value x drive; innate wiring, the gate never rewrites its own ears)
        if P['usVave']:
            # UNCONDITIONED-STIMULUS PATHWAY (innate, non-plastic): visceral/nociceptive US drives
            # the aversive value cells DIRECTLY (amygdala biology: the US reaches CeA/BLA and FIRES
            # it — that is what makes CS->fear learnable at the receiving end). ANATOMICALLY
            # SEPARATE from ncmp: task-error feedback (rWrong) teaches POLICY (D2/melt via 5HT)
            # and must never write option-fear (measured: ncmp->vAVE made every apprentice error
            # a taste aversion against the hardest dish — med-staple attractor in 3/3 seeds).
            conn(B['ucmp'], B['vave'], 0.9, 0.35, 0.60)
        if P['valTD']:   # V2+: the brain computes its own RPE — prediction subtracts from outcome
            conn(B['vapp'], B['pcmp'], 0.9, 0.36, 0.72, -1)   # expected-good reduces positive surprise (gain per V0 measurement)
            conn(B['vave'], B['ncmp'], 0.9, 0.36, 0.72, -1)   # expected-bad no longer re-billed (fear converges)
            if P['tdCross']:
                # CROSS PAIR — DISABLED until comparators get PHASIC physics. As tonic wires they
                # are slow poison (measured stage B): vave->pcmp pays 'relief' DA continuously
                # whenever fear cells fire — fear CONVERTS to appetite (EMPTY topped 3/6 seeds,
                # engage 1.00, self-feeding perpetual motion); vapp->ncmp taxes every meal with
                # appetite-proportional 5HT — rich crushed by its own expectation (-0.21..+0.05).
                # Relief/disappointment are REAL biology but they are TRANSIENTS (omission-onset
                # rebound; LHb phasics), not standing currents. Next knife: comparator adaptation.
                conn(B['vapp'], B['ncmp'], 0.9, 0.36, 0.72)
                conn(B['vave'], B['pcmp'], 0.9, 0.36, 0.72)
        if P['hRace']:
            # ZERO-ORGAN MODE: no volition gate exists; hunger's peptide modulates the RACE
            # itself - a hungry animal's policy is excitable, a sated one is quiet. Same
            # ligand physics as hchan, receptors on the answer populations.
            conn(B['hun'], self.outs, 1.0, P['hW'] * 0.95, P['hW'] * 1.05, 1, 4)
        if P['hchan']:
            # HUNGER LIGAND (stages 2-4): hun cells release H on spikes; gov receptors'
            # occupancy Q[4] locally implements drive-multiplication, bravery, and
            # desperation noise - replacing three host-computed hunger-mean reads.
            conn(B['hun'], B['gov'], 1.0, P['hW'] * 0.95, P['hW'] * 1.05, 1, 4)  # near-uniform broadcast (knife-3 lesson)
        ccs = len(w)
        if P['curioSyn']:
            # STAGE-1 CELLIZATION (Lee 2026-08-11): curiosity as presynaptic habituation.
            # mem->gov excitatory wires whose transmission is scaled by (1 - hab[pre]) each tick:
            # a novel option's memory cells drive the volition gate at full strength; familiar
            # ones fade as their hab trace charges — Aplysia-style habituation as synaptic
            # depression, replacing the host-computed novelty bonus novB. Non-plastic by design:
            # the signal is familiarity, not learned value.
            conn(B['mem'], B['nov'], P['curioPc'], P['curioW'] * 0.5, P['curioW'] * 1.5)
        cce = len(w)
        if P['curioSyn']:
            # DIVISIVE NORMALIZATION (Carandini-Heeger canonical circuit; PV feedforward
            # inhibition): the same memory ensemble that drives the curiosity channel also
            # drives a small inhibitory pool whose shunting conductance divides the channel
            # at each gov cell - ensemble-size invariance, measured disease: unnormalized
            # sum overdelivers 1.3-3.1x with ensemble size (stage1_diag 2026-08-11).
            conn(B['mem'], B['nrm'], 0.5, P['nrmInW'] * 0.8, P['nrmInW'] * 1.2)
            conn(B['nrm'], B['nov'], 1.0, P['nrmOutW'] * 0.8, P['nrmOutW'] * 1.2, -1, -1, False, True)
            # RELAY BROADCAST (knife 3): the pooled, normalized novelty is broadcast to the
            # volition gate diffusely and near-uniformly - the Lisman-Grace topology. Novelty
            # reaches the gate only through the nucleus, never by direct heterogeneous wires
            # (measured disease of knife 1+2: sparse direct wires give a few gov cells excess
            # drive; hungry noise then tips them - 407 vs 87 empty-patch visits, brain 3).
            conn(B['nov'], B['gov'], 1.0, P['novOutW'] * 0.95, P['novOutW'] * 1.05)
        cceAll = len(w)
        for a in B['cohorts']:
            for b in B['cohorts']:
                if a is not b: conn(a, b, 0.35, 0.10, 0.25, -1)            # race lateral inhibition (innate)
        for co in B['cohorts']: conn(co, co, 0.7, 0.25, 0.38)              # self-excitation (innate)
        rce = len(w)
        if P['achWire']:
            # CUE-DRIVEN BASAL-FOREBRAIN ACh (sensory salience line): novel input drives BF
            # through real synapses instead of the protocol injecting ach=0.8 with each cue.
            conn(B['sens'], B['ach'], 0.5, P['achW'] * 0.8, P['achW'] * 1.2)
        conn(B['pcmp'], B['da'], 1.0, 0.40, 0.70); conn(B['ncmp'], B['fht'], 1.0, 0.40, 0.70)
        if P['hasselmo']:
            # HASSELMO WRITE-MODE + HANGYA SALIENCE: outcomes (reward, error, frustration)
            # burst basal-forebrain ACh; plasticity requires ACh occupancy (read/write toggle
            # in chemistry, no boolean anywhere). Cue-onset ACh (sensory salience) already
            # arrives via symExt - encoding mode is stimulus-driven, biologically so.
            conn(B['pcmp'], B['ach'], 1.0, 0.40, 0.70)
            conn(B['ncmp'], B['ach'], 1.0, 0.40, 0.70)
            conn(B['ucmp'], B['ach'], 1.0, 0.40, 0.70)
        if P['uncExplore'] or P['naWire']:
            # NORADRENALINE = UNEXPECTED UNCERTAINTY (Yu & Dayan): any comparator surprise,
            # appetitive or aversive, fires NA — anatomically, not by world injection.
            # |r - V| = pcmp activity + ncmp activity (prediction already subtracted by the
            # TD inhibition wires) — the scripted arousal was a redundant copy of information
            # physically present in the comparators (debt cleared 2026-08-12).
            wNA = P['naW'] if P['naWire'] else 0.475
            conn(B['pcmp'], B['na'], 1.0, wNA * 0.85, wNA * 1.15)
            conn(B['ncmp'], B['na'], 1.0, wNA * 0.85, wNA * 1.15)
            conn(B['ucmp'], B['na'], 1.0, 0.35, 0.60)
        if P['dipWire']:
            conn(B['ncmp'], B['da'], 0.9, 0.40, 0.70, -1)   # LHb->RMTg->DA: bad news DIPS dopamine below tonic
        conn(B['ucmp'], B['fht'], 1.0, 0.40, 0.70); conn(B['ucmp'], B['clr'], 1.0, 0.30, 0.50)  # US: 5HT release + clearance, same physics as ncmp
        conn(B['pcmp'], B['clr'], 1.0, 0.30, 0.50); conn(B['ncmp'], B['clr'], 1.0, 0.30, 0.50)
        tgt = np.concatenate([B['exc'], self.outs, B['mem'], B['gov'], B['vapp'], B['vave']])  # VAL must hear DA/5HT or its learning is exactly zero (measured)
        for ch, nm in enumerate(('da', 'na', 'ach', 'fht')):
            conn(B[nm], tgt, 0.9, 0.03, 0.06, 1, ch)
        self.src = np.array(src, np.int64); self.dst = np.array(dst, np.int64)
        self.w = np.array(w); self.E = len(w)
        self.modc = np.array(modc, np.int64)
        self.dendF = np.array(dend, bool); self.vetoF = np.array(veto, bool)
        self.somaF = ~self.dendF & ~self.vetoF & (self.modc < 0)
        self.modF = self.modc >= 0
        self.d1F = np.zeros(self.E, bool); self.d1F[d1s:d1e] = True
        self.d2F = np.zeros(self.E, bool); self.d2F[d2s:d2e] = True
        self.moF = np.zeros(self.E, bool); self.moF[mos:moe] = True        # mem->outs class
        self.vAppF = np.zeros(self.E, bool); self.vAppF[vps:vpe] = True   # value opponent classes
        self.vAveF = np.zeros(self.E, bool); self.vAveF[vns:vne] = True
        raceF = np.zeros(self.E, bool); raceF[rcs:rce] = True
        self.curioF = np.zeros(self.E, bool); self.curioF[ccs:cce] = True   # habituating curiosity wires
        curioFam = np.zeros(self.E, bool); curioFam[ccs:cceAll] = True      # whole curiosity family incl. nrm
        self.somaF = self.somaF & ~self.curioF                              # habituating wires driven separately
        self.plast = ~self.modF & ~raceF & ~curioFam                        # plastic edges (curiosity family fixed)
        self.plasticIdx = np.nonzero(self.plast)[0]
        self.curioIdx = np.nonzero(self.curioF)[0]
        self.valEdgeIdx = np.nonzero(self.vAppF | self.vAveF)[0]   # Turrigiano scaling targets
        self.vAveIdx = np.nonzero(self.vAveF)[0]                    # fear-ledger edges (relief consolidation)
        self.fearTr = np.zeros(self.N)                              # fear shadow trace (offset detector, intracellular)
        self.vHome = np.full(self.N, P['vSet'])                       # value-cell activity EMA (born at setpoint)
        self.cmpA = np.zeros(self.N)                                  # comparator adaptation (sensor-side annealing)
        # per-preneuron D1/D2 groups for axonal budgets
        self.d1Idx = np.nonzero(self.d1F)[0]; self.d2Idx = np.nonzero(self.d2F)[0]
        # cell classes
        self.isMem = np.zeros(self.N, bool); self.isMem[B['mem']] = True
        self.isRace = np.zeros(self.N, bool); self.isRace[self.outs] = True
        self.cohOf = np.full(self.N, -1, np.int64)
        for c, co in enumerate(B['cohorts']): self.cohOf[co] = c
        self.isPos = np.zeros(self.N, bool); self.isPos[B['posC']] = True
        self.posCtr = np.zeros(self.N); self.posCtr[B['posC']] = np.arange(nPos) / (nPos - 1)
        self.memPos = np.zeros(self.N); self.memPos[B['mem']] = SLOTC[np.arange(nMem) % len(SLOTC)]  # rank cells
        # state
        self.act = np.zeros(self.N); self.spk = np.zeros(self.N, bool)
        self.plat = np.zeros(self.N); self.ah = np.zeros(self.N); self.trigA = np.zeros(self.N)
        self.home = np.full(self.N, 0.08); self.cen = np.zeros(self.N)
        self.goI = np.zeros(self.N); self.adC = np.zeros(nSym)
        self.exp = np.zeros((5, self.N)); self.Q = np.zeros((5, self.N))   # ch4 = hunger ligand H
        self.elig = np.zeros(self.E); self.eligV = np.zeros(self.E); self.hold = np.zeros(self.N)
        self.wv = np.zeros(self.N)                                          # critic (engineering debt, SPEC 1.7)
        self.bufSlow = 0.0; self.pos = SLOTC[0]; self.posOn = False
        # structural turnover state (SPEC 1.12)
        self.turnF = self.d1F | self.moF                                    # excitatory readout classes
        if P['turnD2']: self.turnF |= self.d2F                              # inhibitory: only if enabled
        self.turnIdx = np.nonzero(self.turnF)[0]
        self.use = np.zeros(self.E)                                          # uncentred usage (leaky avg of pre*post)
        self.trials = 0; self.pruned_total = 0
        # Layer 4 consolidation state
        self.prp = np.zeros(self.N)            # cell-wide plasticity-related-protein pool
        self.prot = np.zeros(self.E)           # per-synapse consolidation level
        self.wa = self.w.copy()                # structural anchor (spine scaffold set-point)
        self.creb = np.zeros(self.N)           # per-cell slow nuclear integrator (CREB analog)
        self._val = np.concatenate([blocks['vapp'], blocks['vave']])
        self._cmp = np.concatenate([blocks['pcmp'], blocks['ncmp']])
        # body (physiology, attached to the organism): energy + innate graded hunger tuning
        self.energy = P['Esetpoint']
        self.hunTheta = np.linspace(0.0, 0.4, len(blocks['hun']))            # graded deficit thresholds
        # volition state
        self.isGoV = np.zeros(self.N, bool); self.isGoV[blocks['gov']] = True
        # alliesthesia pathway index: the value->volition edges, gain-gated by interoceptive state
        # (bio: leptin/orexin gating of NAcc incentive salience — wanting = value x drive, not +)
        self.vgIdx = np.nonzero(np.isin(self.src, blocks['vapp']) & np.isin(self.dst, blocks['gov']))[0]
        self.vgAveIdx = np.nonzero(np.isin(self.src, blocks['vave']) & np.isin(self.dst, blocks['gov']))[0]
        self.goVI = np.zeros(self.N)
        # birth baseline of the D2 stopping wall per slot: overcomplete birth gives raw mass at
        # day 0 — stop-confidence must read only the LEARNED excess, or exploration is strangled
        # at birth (measured: TD-world 800 days, 3 seeds, engaged daily, never once answered
        # right, starved at E=0.00 — the first meal never happened).
        d2i0 = self.d2Idx
        self.wallBirth = np.array([float((np.exp(-((self.posCtr[self.src[d2i0]] - sc) ** 2)
                                   / (2 * P['sigmaPosC'] ** 2)) * np.abs(np.minimum(self.w[d2i0], 0))).sum())
                                   for sc in SLOTC])
        self.hab = np.zeros(self.N)   # per-cell familiarity trace (curiosity organ; mem cells only)
        self.unc = np.full(self.N, 0.5)  # per-cell 'not understood' trace (uncertainty-directed exploration): born half-uncertain
        # cached mem block views
        self._mem = B['mem']; self._posC = B['posC']

    # ---------------- tick: the whole physics, one step ----------------
    def tick(self, ext, learn):
        pr = P
        aw = self.w * self.act[self.src]
        inc = np.zeros(self.N); incD = np.zeros(self.N); veto = np.zeros(self.N)
        np.add.at(inc, self.dst[self.somaF | self.modF], aw[self.somaF | self.modF])
        if len(self.curioIdx):
            # curiosity channel: presynaptic habituation scales release (repetition suppression),
            # and the nrm pool's shunting conductance divides it (canonical normalization) —
            # both quantities local to each synapse / each gov cell.
            ci = self.curioIdx
            cbuf = np.zeros(self.N)
            np.add.at(cbuf, self.dst[ci], self.w[ci] * self.act[self.src[ci]] * (1.0 - self.hab[self.src[ci]]))
            nvc = self.blocks['nov']
            inc[nvc] += cbuf[nvc] / (1.0 + pr['kNrm'] * veto[nvc])
        np.add.at(incD, self.dst[self.dendF], aw[self.dendF])
        np.add.at(veto, self.dst[self.vetoF], np.maximum(0, -aw[self.vetoF]))
        if pr['allies'] and pr['needs']:
            # ALLIESTHESIA: satiety multiplicatively devalues incentive at the volition gate.
            # A +0.74 patch value must NOT clear the rheobase on a full stomach (measured:
            # seed3 became a rich-patch specialist, engaging at E=0.95 forever). Additive
            # hun->gov drive stays (bootstrap exploration before any value exists).
            if pr['hchan']:
                q4d = self.Q[4]
                vg = self.vgIdx
                g_loc = pr['h0Want'] + (1.0 - pr['h0Want']) * np.clip(pr['hGain'] * q4d[self.dst[vg]], 0, 1)
                np.add.at(inc, self.dst[vg], -(1.0 - g_loc) * aw[vg])
            else:
                hunM = float(self.act[self.blocks['hun']].mean())
                g = pr['h0Want'] + (1.0 - pr['h0Want']) * hunM
                vg = self.vgIdx
                np.add.at(inc, self.dst[vg], -(1.0 - g) * aw[vg])
            if pr['kBrave'] > 0 and len(self.vgAveIdx):
                # HUNGER BRAVES FEAR (AgRP anxiolysis, Betley/Sternson: starving animals take
                # risks). The fear ear's gain falls with hunger — a wrongly-feared option gets
                # re-tasted in desperation, and TD's big surprise overturns the verdict. Without
                # this, wrong fear is locked forever (seeds 7/11: never re-taste, gate 0.00).
                vga = self.vgAveIdx
                if pr['hchan']:
                    gB_loc = 1.0 - pr['kBrave'] * np.clip(pr['hGain'] * self.Q[4][self.dst[vga]], 0, 1)
                    np.add.at(inc, self.dst[vga], -(1.0 - gB_loc) * aw[vga])
                else:
                    gB = 1.0 - pr['kBrave'] * hunM
                    np.add.at(inc, self.dst[vga], -(1.0 - gB) * aw[vga])
        mem = self._mem
        bufLoad = self.plat[mem].mean()
        self.bufSlow += (bufLoad - self.bufSlow) / pr['tauBuf']
        gDe = pr['gD'] / (1 + pr['kEnc'] * self.bufSlow)
        # mem cells: position-gated induction against position-weighted population mean
        d = self.memPos[mem] - self.pos
        pg = np.exp(-(d * d) / (2 * pr['sigmaPos'] ** 2))
        ddm = gDe * pg * np.clip(incD[mem], -2, 2)
        meanDD = max(0.0, (pg * ddm).sum() / (pg.sum() + 1e-6))
        self.home[mem] += ((self.plat[mem] > 0.4) - self.home[mem]) / pr['tauHome']
        thetaEff = (pr['theta0'] + pr['thetaRel'] * meanDD) * (1 + pr['kHome'] * (self.home[mem] - pr['homeTarget']))
        trig = np.clip((ddm - thetaEff) / pr['scaleD'], 0, 1) / (1 + pr['kVetoI'] * veto[mem])
        self.trigA[:] = 0; self.trigA[mem] = trig
        pm = self.plat[mem]
        pm += -pm / pr['tauP'] + pr['kP'] * trig * (1 - pm) - pr['kVeto'] * veto[mem] * pm - pr['kAHP'] * self.ah[mem] * pm
        self.plat[mem] = np.clip(pm, 0, 1)
        self.ah[mem] += -self.ah[mem] / pr['tauAH'] + self.act[mem]
        eg = pg * pg if pr['exprGate'] else 1.0
        base = pr['gain'] * np.clip(inc, -2, 2) + ext + self.hold
        newact = np.clip(np.tanh(base), 0, 1)
        newact[mem] = np.clip(np.tanh(base[mem] + pr['gP'] * self.plat[mem] * eg - pr['gA'] * self.ah[mem]), 0, 1)
        race = self.outs
        raceGain = 1.0
        if pr['hRace']:
            # hunger's peptide scales race drive: g = h0 + (1-h0)*occupancy (per race cell)
            raceGain = pr['h0Want'] + (1.0 - pr['h0Want']) * np.clip(pr['hGain'] * self.Q[4][race], 0, 1)
        self.goI[race] += (-self.goI[race] / pr['tauAcc'] + raceGain * pr['kGo'] * np.clip(inc[race], -2, 2)
                           + self.hold[race] - pr['LEAK'] - pr['gAd'] * self.adC[self.cohOf[race]])
        newact[race] = np.clip(np.tanh(self.goI[race]), 0, 1)
        gv = self.blocks['gov']   # volition accumulator: confidence drive vs intrinsic rheobase leak
        self.goVI[gv] += (-self.goVI[gv] / pr['tauAccV'] + pr['kGoV'] * np.clip(inc[gv], -2, 2)
                          + self.hold[gv] - pr['LEAKV'])
        newact[gv] = np.clip(np.tanh(self.goVI[gv]), 0, 1)
        vl = self._val   # SPN value cells: silent at rest, up-state only on constellation match
        newact[vl] = np.clip(np.tanh(pr['gVal'] * (np.clip(inc[vl], -2, 2) - pr['thetaUp'])), 0, 1)
        if pr['needs']:
            self.energy = max(0.0, self.energy - 1.0 / pr['tauMetab'])       # metabolic burn
            deficit = max(pr['Esetpoint'] - self.energy, 0.0)
            hu = self.blocks['hun']                                          # innate interoceptive tuning
            newact[hu] = np.clip(np.tanh(4.0 * (deficit - self.hunTheta)), 0, 1)
        if self.posOn:
            pc = self._posC
            dpc = self.posCtr[pc] - self.pos
            pd = pr['posAmp'] * np.exp(-(dpc * dpc) / (2 * pr['sigmaPosC'] ** 2))
            newact[pc] = np.clip(np.tanh(pr['gain'] * np.clip(inc[pc], -2, 2) + pd + ext[pc]), 0, 1)
        if pr['tonicMod']:
            for nm, lvl in (('da', pr['tonicDA']), ('na', pr['tonicNA']),
                            ('ach', pr['tonicACH']), ('fht', pr['tonicHT'])):
                c = self.blocks[nm]
                newact[c] = np.clip(np.tanh(pr['gain'] * np.clip(inc[c], -2, 2) + lvl
                                            + pr['sigTonic'] * self.rng.standard_normal(len(c))), 0, 1)
        if pr['sigBg'] > 0:
            bg = np.concatenate([self.blocks['exc'], self._mem])
            newact[bg] = np.clip(newact[bg] + pr['sigBg'] * self.rng.standard_normal(len(bg)), 0, 1)
        self.act = newact
        if pr['cmpAdapt']:
            # SENSOR-SIDE ANNEALING (receptor desensitization at the comparators): a repeated,
            # identical outcome excites its comparator less and less; change restores response.
            # The learning machine's teaching anneals without any predictor - the fatigue lives
            # in the sensor. (Biologize-stack medicine, live-validated in the elder system.)
            cmp_cells = self._cmp
            self.act[cmp_cells] = self.act[cmp_cells] / (1.0 + pr['kCmpA'] * self.cmpA[cmp_cells])
            self.cmpA[cmp_cells] += (self.act[cmp_cells] - self.cmpA[cmp_cells]) / pr['tauCmpA']
        self.spk = self.act > self.thr
        self.ageT = getattr(self, 'ageT', 0) + 1     # developmental clock (ticks since birth)
        self.adC *= (1 - 1 / pr['tauAd'])
        if pr['curio'] or pr['curioSyn']: self.hab *= (1 - 1 / pr['tauHab'])
        if pr['reliefCell']:
            # RELIEF AS A CELL RULE (offset response): each fear cell carries a shadow of its
            # own recent activity; the moment activity falls below the shadow - fear ebbing -
            # the synapses that carried that fear capture protection, in proportion to the ebb.
            # No event hook, no host detection: the cells feel the relief themselves.
            vav = self.blocks['vave']
            ebb = np.maximum(self.fearTr[vav] - self.act[vav], 0)
            self.fearTr[vav] += (self.act[vav] - self.fearTr[vav]) / pr['tauFearTr']
            if ebb.max() > 1e-4:
                ebbN = np.zeros(self.N); ebbN[vav] = ebb
                vi = self.vAveIdx
                inc_p = pr['kRelCell'] * self.act[self.src[vi]] * ebbN[self.dst[vi]]
                grew = inc_p > 1e-5
                if grew.any():
                    self.prot[vi] += inc_p * (1 - self.prot[vi])
                    wav = self.wa[vi]; wav[grew] = self.w[vi][grew]; self.wa[vi] = wav
        if pr['vScale']:
            # TURRIGIANO SYNAPTIC SCALING on value cells (Turrigiano 1998; ref [7]): a value
            # population whose slow activity average falls below its setpoint multiplicatively
            # upscales its afferents until it speaks again. Silence is corrected, structurally -
            # the anti-death-spiral floor biology gives every newborn appetite system. Pure
            # maintenance (reads only own activity), never surprise-fed (First Law compliant).
            vc = self._val
            self.vHome[vc] += (self.act[vc] - self.vHome[vc]) / pr['tauVH']
            defi = pr['vSet'] - self.vHome
            vi = self.valEdgeIdx
            d = defi[self.dst[vi]]
            up = d > 0
            if up.any():
                self.w[vi[up]] = np.minimum(self.w[vi[up]] * (1.0 + pr['kVScale'] * d[up]), 1.5)
        # modulators: spike-contingent release, receptor occupancy
        self.exp *= np.exp(-1 / pr['tauX'])
        if pr['datPump']:
            # DAT/SERT reuptake: ACTIVE transmitter clearance. During a teach window the
            # comparators keep firing and re-supplying ligand, so in-window occupancy holds;
            # once release stops, the tail dies in ~1-2 ticks instead of ~10 - the chemistry
            # that lets biology keep plasticity always-on without a teaching boolean.
            self.exp[:4] *= np.exp(-1 / pr['tauDAT'])
        rel = self.modF & self.spk[self.src]
        if rel.any():
            amt = self.w[rel].copy()
            ch4 = self.modc[rel] == 4
            if ch4.any():
                amt[ch4] *= self.act[self.src[rel][ch4]]   # H: rate-scaled release (peptide tone)
            np.add.at(self.exp, (self.modc[rel], self.dst[rel]), amt)
        self.Q = self.exp / (pr['Khalf'] + np.abs(self.exp))
        if pr['uncExplore']:
            m = self._mem
            am = self.act[m]; qna = self.Q[1][m]
            # surprise (NA present) makes an experienced option LESS understood; quiet, well-predicted
            # experience SETTLES it; disuse drifts everything slowly back toward uncertain.
            self.unc[m] += (pr['kUncUp'] * am * qna * (1 - self.unc[m])
                            - pr['kSettle'] * am * (1 - qna) * self.unc[m]
                            + (1 - self.unc[m]) / pr['tauStale'])
            np.clip(self.unc[m], 0, 1, out=self.unc[m])
        # region-mean centring
        rs = np.bincount(self.region, self.act, self.nreg); rc = np.bincount(self.region, None, self.nreg)
        self.cen = self.act - rs[self.region] / np.maximum(1, rc[self.region])
        # dendritic refinement (SPEC 1.3-5)
        de = self.dendF
        tg = self.trigA[self.dst[de]]; pa = self.act[self.src[de]]; wd = self.w[de]
        ltp = (tg > 0.15) & (pa > 0.5); ltd = (tg > 0.45) & (pa < 0.10)
        wd[ltp] += pr['etaDendLTP'] * tg[ltp] * (pr['wDendMax'] - wd[ltp])
        wd[ltd] -= pr['etaDendLTD'] * tg[ltd] * wd[ltd]
        self.w[de] = wd
        # usage trace for turnover classes (uncentred by design: survives uniform regions, SPEC 1.12)
        if P['turnover']:
            ti = self.turnIdx
            ap = self.act[self.src[ti]]; aq = self.act[self.dst[ti]]
            prod = ap * aq
            if P['turnD2'] == 2:                                             # inverted usage for inhibition
                m = self.d2F[ti]
                prod[m] = ap[m] * (1 - aq[m])
            if P['useGate']:                                                 # activity-gated: visit-rate invariant
                g = np.clip(ap + aq, 0, 1)
                self.use[ti] += g * (prod - self.use[ti]) / P['tauUse']
            else:
                self.use[ti] += (-self.use[ti] + prod) / P['tauUse']
        # eligibility with presynaptic-release gate
        pi = self.plasticIdx
        gate = self.spk[self.src[pi]]
        prod = self.cen[self.src[pi]] * self.cen[self.dst[pi]]
        self.elig[pi] += -self.elig[pi] / pr['tauE'] + np.where(gate, prod, 0)
        if pr['valSlowE']:
            # DUAL ELIGIBILITY TIMESCALES (Yagishita 2014: SPN value-spine eligibility ~1-2s,
            # motor/cortical traces sub-second). Value synapses read a SLOW trace: a modulator
            # arriving after a gap finds motor eligibility dead but value eligibility alive —
            # delayed outcomes (frustration, late meals) write onto OPTIONS, never onto ACTIONS.
            # (Measured: immediate frustration corroded skill 1.00->0.83; cue-re-presentation
            # re-energized the policy chain and corroded it to 0.25.)
            self.eligV[pi] += -self.eligV[pi] / pr['tauEV'] + np.where(gate, prod, 0)
        # Layer 4: protein synthesis (cell-wide, salience-gated, every tick) + anchor spring
        if pr['anchor']:
            if pr['crebGate']:  # slow nuclear integrator: kinase (DA+NA) vs phosphatase (5HT), activity-gated
                drive = self.act * (self.Q[0] + self.Q[1]) - pr['kPhos'] * self.act * self.Q[3]
                if pr['fearAnchor']:
                    # fear cells' CREB is keyed to their own ligand: 5-HT charges, DA dephosphorylates
                    vav = self.blocks['vave']
                    drive[vav] = self.act[vav] * (self.Q[3][vav] + self.Q[1][vav]) - pr['kPhos'] * self.act[vav] * self.Q[0][vav]
                self.creb += (pr['kCreb'] * drive - self.creb) / pr['tauCREB']
                np.clip(self.creb, 0, 2, out=self.creb)
            else:
                sal = np.abs(self.Q[0] - self.Q[3]) * (1 + self.Q[1] + self.Q[2])
                self.prp += -self.prp / pr['tauPRP'] + pr['kPRP'] * np.clip((sal - pr['thetaPRP']) / pr['scalePRP'], 0, 1) * (1 - self.prp)
            # MAINTENANCE IS STRUCTURAL: DA gates induction (creb/capture), never upkeep.
            # prot pays rent only in its own active context, and slowly (late-LTP is protein/actin
            # scaffold, not dopamine-fed; a habit is precisely a policy that survives zero surprise).
            paP = self.act[self.src[pi]]
            self.prot[pi] -= self.prot[pi] / pr['tauProt'] * paP
            self.w[pi] += pr['kAnc'] * self.prot[pi] * (self.wa[pi] - self.w[pi])   # elastic restoration
        if learn or P['alwaysOn']:
            dsts = self.dst[pi]
            Mr = self.Q[0, dsts] - self.Q[3, dsts]
            if pr['hasselmo']:
                # ACh as the write-enable: no ligand, no plasticity (retrieval mode);
                # occupancy saturates Michaelis-style toward full write (encoding mode).
                achG = self.Q[2, dsts] / (pr['thetaACh'] + self.Q[2, dsts])
                U = achG * (1 + self.Q[1, dsts])
                lamG = achG
            else:
                U = 1 + self.Q[1, dsts] + self.Q[2, dsts]
                lamG = np.ones(len(dsts))
            base_dw = pr['eta'] * U * self.elig[pi]
            if pr['anchor']:   # consistency-gated capture (reconsolidation): grow on rewarded
                # participation, DESTABILIZE on punished participation.
                ae = np.abs(self.elig[pi])
                if pr['crebGate']:
                    # CREB gate: capture only where the postsynaptic cell's SLOW success integral
                    # has crossed threshold — and the anchor SNAPS to the (by-then converged) weight.
                    gate = self.creb[dsts] > pr['thetaCREB']
                    if pr['fearAnchor']:
                        # EACH LEDGER CONSOLIDATES UNDER ITS OWN TEACHING SIGN (amygdala
                        # consolidation is aversive-keyed): appetitive edges capture on DA
                        # verdicts, fear edges on 5-HT verdicts - and neither organ's lessons
                        # melt its own anchors. Cures the one-signed anchor that made learned
                        # fear structurally un-consolidatable (rent forever wins; C5 limit cycle).
                        sgn = np.where(self.vAveF[pi], -1.0, 1.0)
                        Mre = Mr * sgn
                        cap = pr['kCap'] * ae * gate * (Mre > 0)
                        melt = pr['kMelt'] * ae * np.maximum(-Mre, 0)
                    else:
                        cap = pr['kCap'] * ae * gate * (Mr > 0)
                        melt = pr['kMelt'] * ae * np.maximum(-Mr, 0)
                    self.prot[pi] += cap * (1 - self.prot[pi]) - melt * self.prot[pi]
                    snap = cap > 0
                    if snap.any():
                        waopi = self.wa[pi]; waopi[snap] = self.w[pi][snap]; self.wa[pi] = waopi
                else:
                    cap = pr['kCap'] * ae * self.prp[dsts] * (Mr > 0)
                    melt = pr['kMelt'] * ae * np.maximum(-Mr, 0)
                    self.prot[pi] += cap * (1 - self.prot[pi]) - melt * self.prot[pi]
                    self.wa[pi] += pr['kWa'] * cap * (self.w[pi] - self.wa[pi])
                    np.subtract.at(self.prp, dsts, pr['kDrain'] * cap)               # finite protein: competition
                    np.clip(self.prp, 0, 1, out=self.prp)
            wpi = self.w[pi]
            d1 = self.d1F[pi]; d2 = self.d2F[pi]
            vap = self.vAppF[pi]; vav = self.vAveF[pi]
            raceDst = self.isRace[dsts] & ~d1 & ~d2
            other = ~d1 & ~d2 & ~raceDst & ~vap & ~vav
            # policy decay is ACTIVITY-GATED (same law as value decay, knife 5): a wall/map pays
            # rent only when its own presynaptic context is present — other slots' teach ticks no
            # longer grind it. Measured: closed-loop D2 wall built to 3.71 then evaporated to 0.27
            # once predicted punishment cut its income while global rent kept charging.
            preAP = self.act[self.src[pi]]
            wpi[d1] = np.clip(wpi[d1] + base_dw[d1] * np.maximum(Mr[d1], 0) - pr['lamD1'] * pr['lamScale'] * lamG[d1] * wpi[d1] * preAP[d1], -0.2, 1.5)
            wpi[d2] = np.clip(wpi[d2] + pr['etaD2fac'] * base_dw[d2] * np.minimum(Mr[d2], 0) - pr['lamD2'] * pr['lamScale'] * lamG[d2] * wpi[d2] * preAP[d2], -1.5, 0.05)
            if pr['valcells']:  # opponent value learning (ventral striatum D1/D2 family):
                # vAPP grows on under-predicted reward, vAVE on under-predicted punishment;
                # downward value corrections come from the opponent growing, not from depression.
                # Decay is ACTIVITY-GATED (pre present = state visited): wall-clock decay loses to
                # visit-rate arithmetic at any curriculum size (measured: strong winners p95 1.40
                # ground to 0.06 by ep300). Depotentiation needs activity; disused memory holds.
                # SIGNED TD on opponent channels (bio: D1 = burst-LTP AND dip-LTD; D2 mirror).
                # Rectified grow-only channels integrate SURPRISE, not value: once predictions
                # converge M+ vanishes, vAPP starves while residual errors keep feeding vAVE ->
                # measured near-perfect anticorrelation (-0.995). Signed delta freezes at delta=0:
                # confirmed value pays no rent. valLam kept tiny, activity-gated (junk cleanup only).
                preA = preAP
                # FIRST LAW, evidence gate (5th appearance): appetitive value may only GROW on
                # positive evidence (Mr>0), aversive value only on negative evidence (Mr<0).
                # Without this, the opponent mirror's algebra lets elig<0 x Mr>0 write FEAR from
                # REWARD — and the 0.1x extinction asymmetry ratchets it: every meal pumps vAVE
                # at full speed, drains at 0.1x. Measured (V4 ledger): patch negativity tracks
                # meal count with ZERO aversive events; rich -0.75, never-fed EMPTY ~0.
                base_v = (pr['eta'] * U * self.eligV[pi]) if pr['valSlowE'] else base_dw
                dva = pr['valEta'] * base_v[vap] * Mr[vap]
                if pr['valEvid']:
                    dva = np.where((dva > 0) & (Mr[vap] <= 0), 0.0, dva)
                wpi[vap] = np.clip(wpi[vap] + dva - pr['valLam'] * pr['lamScale'] * lamG[vap] * wpi[vap] * preA[vap], 0.0, 1.5)
                # aversion extinguishes slowly (amygdala-style): the M+ direction (unlearning
                # fear) runs at 0.1x — otherwise closed-loop TD re-demolishes the stopping wall
                # on silent visits (diag11's disease, in-brain edition).
                dwv = -pr['valEta'] * base_v[vav] * Mr[vav]
                if pr['matAve'] > 0:
                    # AVERSIVE LATE MATURATION (Sullivan sensitive period): the fear ledger's
                    # write pathway ramps in over infancy - approach learns first, guaranteed
                    # by developmental timing, not by supervision. Intrinsic genetic clock.
                    dwv = dwv * min(1.0, self.ageT / pr['matAve'])
                if pr['valEvid']:
                    dwv = np.where((dwv > 0) & (Mr[vav] >= 0), 0.0, dwv)
                if pr['extinctEvid']:
                    # COUNTERCONDITIONING (bio: DA-burst-gated amygdala depotentiation): fear
                    # extinction speed scales with the POSITIVE EVIDENCE present — a strong
                    # unexpected reward at a feared cue reverses fear near full rate; weak
                    # relief (future valTD stopping-wall case) still extinguishes slowly.
                    # Without this, early-learning errors fossilize at 0.1x: 600 perfect meals
                    # cannot erase 30 apprentice stumbles (measured: V1b slot0 all-negative at
                    # v_host +1.0; V4 rich most-negative BECAUSE most-practiced).
                    fac = 0.1 + 0.9 * np.clip(Mr[vav] / pr['extinctRef'], 0.0, 1.0)
                    dwv = np.where(dwv < 0, dwv * fac, dwv)
                else:
                    dwv = np.where(dwv < 0, dwv * 0.1, dwv)
                wpi[vav] = np.clip(wpi[vav] + dwv - pr['valLam'] * pr['lamScale'] * lamG[vav] * wpi[vav] * preA[vav], 0.0, 1.5)
            wpi[raceDst] = np.clip(wpi[raceDst] + base_dw[raceDst] * Mr[raceDst], -pr['wCapRace'], pr['wCapRace'])
            wpi[other] = np.clip(wpi[other] + base_dw[other] * Mr[other], -1.5, 1.5)
            self.w[pi] = wpi
            # D1/D2 axonal output budgets (engineering debt, SPEC 1.6)
            for idx, budget in ((self.d1Idx, pr['D1budget']), (self.d2Idx, pr['D2budget'])):
                s = np.abs(self.w[idx])
                tot = np.bincount(self.src[idx], s, self.N)
                over = tot > budget
                if over.any():
                    f = np.ones(self.N); f[over] = budget / tot[over]
                    self.w[idx] *= f[self.src[idx]]

    # ---------------- protocol layer (world) ----------------
    def sym_ext(self, s, ach=True):
        ext = np.zeros(self.N)
        ext[self.blocks['sens'][s * self.cps:(s + 1) * self.cps]] = 1.0
        if ach and not P['achWire']: ext[self.blocks['ach']] = 0.8
        return ext

    def cohort_mean(self, c): return self.act[self.blocks['cohorts'][c]].mean()

    def trial_reset(self):
        self.plat[:] = 0; self.ah[:] = 0; self.bufSlow = 0; self.elig[:] = 0; self.eligV[:] = 0
        self.hold[:] = 0; self.goI[:] = 0; self.adC[:] = 0
        self.posOn = True; self.pos = SLOTC[0]

    def encode_seq(self, seq):
        z = np.zeros(self.N)
        for k, s in enumerate(seq):
            self.pos = SLOTC[k]
            ext = self.sym_ext(s)
            for _ in range(P['CUE']): self.tick(ext, False)
            for _ in range(P['GAP']): self.tick(z, False)

    def audit_reward_afferents(self):
        """Structural anti-wireheading audit: enumerate every synaptic source population feeding
        the comparators. The guard IS the absence of wires: no cortical/motor/self-evaluative
        source may appear here. (valueClass in structural form.)"""
        names = {}
        for nm, cells in self.blocks.items():
            if nm == 'cohorts':
                for i, co in enumerate(cells): names.update({c: f'cohort{i}' for c in co})
            else:
                arr = cells if not isinstance(cells, list) else []
                names.update({c: nm for c in np.atleast_1d(arr)})
        srcs = set()
        cmp_cells = set(np.concatenate([self.blocks['pcmp'], self.blocks['ncmp'], self.blocks['ucmp']]))
        for e in range(self.E):
            if self.dst[e] in cmp_cells:
                srcs.add(names.get(int(self.src[e]), f'cell{self.src[e]}'))
        return sorted(srcs)

    def eat(self, size=None):
        """Real food -> energy rises -> the delta IS the primary reward (labeled line).
        Returns r_body. The ONLY positive-reward entry point in the needs era."""
        e0 = self.energy
        self.energy = min(1.0, self.energy + (P['mealSize'] if size is None else size))
        return P['kMeal'] * (self.energy - e0)

    def value_cells(self):
        """VAL organ readout: opponent difference of the value populations."""
        return float(P['kVal'] * (self.act[self.blocks['vapp']].mean() - self.act[self.blocks['vave']].mean()))

    def critic(self):
        return float(self.wv[self._mem] @ self.act[self._mem] + self.wv[self._posC] @ self.act[self._posC])

    def critic_learn(self, r, eta_scale=1.0):
        err = r - self.critic()
        for cells in (self._mem, self._posC):
            self.wv[cells] += P['etaV'] * eta_scale * self.act[cells] * err
        return err

    def engage(self, ext):
        """Volition: deliberation window. The answer populations run under their normal exploration
        noise (as in a race, but nothing is emitted); the GO accumulator listens to them —
        confidence = how strongly the content drives an answer. Crossing = she engages; no crossing
        = she declines (silence). Paused (no ticks) = mute by construction."""
        gv = self.blocks['gov']
        self.goVI[gv] = 0.0
        self.goI[self.outs] = np.minimum(self.goI[self.outs], 0)
        self.pos = SLOTC[0]
        decg = np.exp(-1 / P['tauGoV']); kng = np.sqrt(1 - decg * decg)
        deco = np.exp(-1 / P['tauOU']); kno = np.sqrt(1 - deco * deco)
        # DESPERATION AROUSAL: hunger raises gate noise (AgRP -> foraging arousal). A broke,
        # starving creature must re-bootstrap by stochastic sampling — never by deterministic
        # all-engage (hunGovW keeps drive sub-rheobase) and never by lying still until death
        # (measured: seed23 E=0.00 flatline 200 days once rent outpaced her thin income).
        if P['hchan'] and P['needs']:
            sigV = P['sigmaGoV'] * (1.0 + P['kDespV'] * np.clip(P['hGain'] * self.Q[4][gv], 0, 1))  # per-cell receptor read + cascade gain
        else:
            sigV = P['sigmaGoV'] * ((1.0 + P['kDespV'] * float(self.act[self.blocks['hun']].mean())) if P['needs'] else 1.0)
        novB = 0.0
        if P['uncExplore']:
            mcells = self._mem
            mact = self.act[mcells]
            wsum = float(mact.sum()) + 1e-9
            uncRead = float((self.unc[mcells] * mact).sum()) / wsum
            novB = P['kCurio'] * uncRead
        elif P['curio']:
            # CURIOSITY / UNCERTAINTY SAMPLING (hippocampal novelty -> ACh -> exploration; foraging
            # theory's sampling behavior): options not engaged for a long time regain sampling
            # priority. Without this, avoidance is an ABSORBING state — no visits, no evidence,
            # the First Law freezes stale pessimism forever (measured: staple lottery, 6-seed run:
            # every seed's early-luck staple locked in; rich wins the lottery only ~1/3).
            mcells = self._mem
            mact = self.act[mcells]
            wsum = mact.sum() + 1e-9
            fam = float((self.hab[mcells] * mact).sum() / wsum)   # familiarity of the held option
            novB = P['kCurio'] * (1.0 - fam)
        nzg = sigV * self.rng.standard_normal(len(gv))
        nzo = P['sigmaCoh'] * self.rng.standard_normal(self.nSym)
        for _ in range(P['VMAXW']):
            nzg = nzg * decg + sigV * kng * self.rng.standard_normal(len(gv))
            nzo = nzo * deco + P['sigmaCoh'] * kno * self.rng.standard_normal(self.nSym)
            self.hold[gv] = nzg + novB
            self.hold[self.outs] = nzo[self.cohOf[self.outs]] + P['sigmaOut'] * self.rng.standard_normal(len(self.outs))
            self.tick(ext, False)
            if self.act[gv].mean() > P['THETA_GOV']:
                self.hold[gv] = 0.0; self.hold[self.outs] = 0.0
                self.goI[self.outs] = np.minimum(self.goI[self.outs], 0)   # deliberation leaves no head start
                if P['curio'] or P['curioSyn']:   # engagement habituates the held option's cells; disuse recovers slowly
                    m = self._mem
                    self.hab[m] += P['habUp'] * self.act[m] * (1 - self.hab[m])
                return True
        self.hold[gv] = 0.0; self.hold[self.outs] = 0.0
        self.goI[self.outs] = np.minimum(self.goI[self.outs], 0)
        if P['reliefCap']:
            # RELIEF PAYS THE 'NO' (Mowrer two-factor; safety-signal-gated amygdala
            # consolidation): declining while fear is active is the avoidance circuit
            # WORKING - the phasic relief charges the protection of exactly the fear
            # synapses just consulted. Relief writes anchor, never value: no appetite,
            # no dopamine, nothing to wirehead. Useful 'no' consolidates; wrong 'no'
            # still extinguishes via dopamine counterconditioning when re-tasted.
            # proportional pay, no threshold cliff (Mowrer: relief scales with the fear
            # relieved; a thin fear earns a thin wage - the loop can ignite from any seed).
            # The per-edge 1e-4 filter below is the only noise guard.
            if True:
                vi = self.vAveIdx
                inc_p = P['kRelCap'] * self.act[self.src[vi]] * self.act[self.dst[vi]]
                grew = inc_p > 1e-4
                if grew.any():
                    self.prot[vi] += inc_p * (1 - self.prot[vi])
                    wav = self.wa[vi]; wav[grew] = self.w[vi][grew]; self.wa[vi] = wav
        return False

    def gov_drive(self, reps=30):
        """Instrument: mean synaptic drive into GO cells during a deliberation window (for LEAKV
        calibration: set the rheobase between untrained and trained drive — measure, don't assume)."""
        gv = self.blocks['gov']
        tot = 0.0
        for t in range(reps):
            aw = self.w * self.act[self.src]
            inc = np.zeros(self.N); np.add.at(inc, self.dst[self.somaF], aw[self.somaF])
            tot += P['kGoV'] * np.clip(inc[gv], -2, 2).mean()
            self.tick(np.zeros(self.N), False)
        return tot / reps

    def emit_slots(self, target, max_slots, learn, rScale=1.0):
        z = np.zeros(self.N); emitted = []
        rng = self.rng
        for k in range(max_slots):
            self.pos = SLOTC[k]
            self.goI[self.outs] = np.minimum(self.goI[self.outs], 0)
            self.hold[:] = 0; self.tick(z, False)
            if P['valTD']:
                v = P['kCert'] * self.value_cells()
            elif P['wallCert']:
                # SUBSTRATE CERTAINTY = STRUCTURE (Lee's no-predictor ruling): confidence is
                # carved D2 wall mass beyond birth baseline - 'I have built here' - a local
                # structural read, no predictor anywhere. Early: no walls, full exploration
                # noise; learned: walls annealed the noise. Same organ the valTD era proved.
                v = 0.0
            else:
                v = self.critic()
            # kCert: VAL's readout scale under memOnly+gVal1.5 physics converges ~0.26 where the
            # host critic read ~1.0 (measured stage A2/A3).
            cert = v if v >= 0 else -P['kNegSlope'] * v
            if (P['valTD'] or P['wallCert']) and P['kWallC'] > 0:
                # TWO CONFIDENCES, TWO ORGANS (the memOnly fork completed): VAL (content) carries
                # 'what to say' confidence; the D2 stopping wall (position) carries 'nothing to
                # say here' confidence. Content-only VAL reads 0 beyond end -> sig maxes -> noise
                # (sigma 0.30) overpowers the wall's -0.07/tick suppression and babble never dies
                # (measured A4: 2800 eps, wall mass 7.45 at slot3, acc oscillates 0.0-0.7 forever).
                d2i = self.d2Idx
                pgw = np.exp(-((self.posCtr[self.src[d2i]] - self.pos) ** 2) / (2 * P['sigmaPosC'] ** 2))
                wall = float((pgw * np.abs(np.minimum(self.w[d2i], 0))).sum())
                kslot = int(np.argmin(np.abs(SLOTC - self.pos)))
                cert = max(cert, P['kWallC'] * max(0.0, wall - self.wallBirth[kslot]))
            floor = P['sigFloor']
            if P['needs']:   # hunger keeps exploration alive: a starving creature may not go quiet
                floor = max(floor, P['kHunSig'] * float(self.act[self.blocks['hun']].mean()))
            sig = P['sigmaCoh'] * np.clip(1 - cert, floor, 1)
            dec = np.exp(-1 / P['tauOU']); kn = np.sqrt(1 - dec * dec)
            nz = sig * rng.standard_normal(self.nSym)
            crossed = -1
            for _ in range(P['MAXW']):
                nz = nz * dec + sig * kn * rng.standard_normal(self.nSym)
                self.hold[self.outs] = nz[self.cohOf[self.outs]] + P['sigmaOut'] * rng.standard_normal(len(self.outs))
                self.tick(z, False)
                cms = [self.cohort_mean(c) for c in range(self.nSym)]
                mx = int(np.argmax(cms))
                if cms[mx] > P['THETA_EMIT']: crossed = mx; break
            if crossed < 0:
                emitted.append(-1)
                # silence = no outcome = weak evidence. Recovery toward 0 is GRADED BY AVERSION
                # DEPTH: mild aversion (wrong-symbol -0.25) relaxes fast -> exploration returns
                # (slot rescue); deep aversion (overrun -1) is sticky -> the stopping wall persists.
                # Biology: strong aversive memories resist extinction, mild ones fade.
                if learn and not P['rDirect']:
                    if P['extinctAsym']:
                        vv = (P['kCert'] * self.value_cells()) if P['criticOff'] else self.critic()
                        if vv < 0:   # aversion: stickiness graded by DEPTH (deep wall persists, mild fades)
                            scale = float(np.clip(1 + vv, 0.05, 1.0)) * 0.5
                        else:        # optimism: a good state gone silent cools fast -> rescue returns
                            scale = 0.5
                        if not P['criticOff']: self.critic_learn(0, eta_scale=scale)
                    else:
                        if not P['criticOff']: self.critic_learn(0)
                continue
            emitted.append(crossed)
            if P['needs']: self.energy = max(0.0, self.energy - P['costEmit'])  # effort costs energy
            self.hold[self.outs] = 0
            for _ in range(P['postCross']): self.tick(z, False)
            if learn:
                r = (rScale if crossed == target[k] else P['rWrong']) if k < len(target) else P['rOver']
                if P['valTD']:
                    # the brain computes its own RPE: world injects RAW outcome; the value organ's
                    # synapses subtract the prediction at the comparators (striosome/LHb circuit).
                    if not P['criticOff']: self.critic_learn(r)   # host critic: logging only; dead when criticOff
                    vv = self.value_cells()
                    cons = z.copy()
                    cons[self.blocks['pcmp']] = max(r, 0)
                    cons[self.blocks['ncmp']] = max(-r, 0)
                    if not P['naWire']: cons[self.blocks['na']] = abs(r - vv)   # debt: redundant copy of pcmp+ncmp
                elif P['rDirect']:
                    # LEARNING MACHINE = RAW OUTCOME EVENTS (Lee's ruling 2026-08-12): the
                    # substrate owns no predictor, so its teaching is the world's outcome
                    # itself - correct feeds the surprise comparator, error the disappointment
                    # comparator, at outcome magnitude. Convergence is the CREB consistency
                    # gate's job, not an expectancy subtraction the tissue cannot host.
                    cons = z.copy()
                    cons[self.blocks['pcmp']] = max(r, 0)
                    cons[self.blocks['ncmp']] = max(-r, 0)
                    if not P['naWire']: cons[self.blocks['na']] = abs(r)
                else:
                    rpe = self.critic_learn(r)
                    cons = z.copy()
                    cons[self.blocks['pcmp']] = abs(rpe) if rpe >= 0 else 0
                    cons[self.blocks['ncmp']] = abs(rpe) if rpe < 0 else 0
                    if not P['naWire']: cons[self.blocks['na']] = abs(rpe)
                for _ in range(P['TEACH']): self.tick(cons, True)
            self.adC[crossed] += 1.0
            self.goI[self.blocks['cohorts'][crossed]] = -0.6
            for _ in range(P['slotGap']): self.tick(z, False)
        self.posOn = False
        for _ in range(P['ITI']): self.tick(z, False)
        return emitted

    def ask(self, in_seq, max_slots=5):
        """Her emitted sequence; None = she DECLINED to engage (volition gate, if enabled)."""
        self.trial_reset(); self.encode_seq(in_seq)
        z = np.zeros(self.N)
        for _ in range(4): self.tick(z, False)
        if P['volition'] and not self.engage(z):
            self.posOn = False
            for _ in range(P['ITI']): self.tick(z, False)
            return None
        em = self.emit_slots([], max_slots, False)
        out = []
        for x in em:
            if x < 0: break
            out.append(x)
        return out

    def prune_sweep(self):
        """Activity-dependent pruning + rewiring: weak AND unused synapses in the turnover economy
        get their slot rewired to a fresh random presynaptic cell of the same source region.
        Synapse count conserved (slot rewiring, not neurogenesis). Local criteria only."""
        rng = self.rng
        cand = self.turnF & (np.abs(self.w) < P['wPrune']) & (self.use < P['uPrune'])
        if P['anchor']: cand &= self.prot < P['protSafe']        # consolidated synapses are prune-immune
        idx = np.nonzero(cand)[0]
        if len(idx) == 0: return 0
        cap = max(1, int(P['pruneFrac'] * len(self.turnIdx)))
        if len(idx) > cap: idx = rng.choice(idx, cap, replace=False)
        n = len(idx)
        # rewire per class: new random pre from the class's source region
        sb = P['silentBirth']
        for mask, pool, lo, hi, sign in ((self.d1F, self._posC, 0.02, 0.10, 1),
                                          (self.d2F, self._posC, 0.01, 0.05, -1),
                                          (self.moF, self._mem, 0.02, 0.10, 1)):
            sub = idx[mask[idx]]
            if len(sub) == 0: continue
            self.src[sub] = rng.choice(pool, len(sub))
            if sb:  # nascent synapses born near-silent (AMPA-silent), unsilenced only by LTP capture
                self.w[sub] = sign * rng.uniform(P['nbLo'], P['nbHi'], len(sub))
            else:
                self.w[sub] = sign * rng.uniform(lo, hi, len(sub))
            self.use[sub] = P['uPrune'] * 3          # grace period: newborn not instantly re-pruned
            self.elig[sub] = 0.0; self.eligV[sub] = 0.0
        self.pruned_total += n
        return n

    def train_trial(self, in_seq, out_seq):
        self.trial_reset(); self.encode_seq(in_seq)
        z = np.zeros(self.N)
        for _ in range(4): self.tick(z, False)
        engaged = True
        if P['volition'] and not self.engage(z):
            engaged = False           # she declined: no action -> no outcome -> no teaching
            self.posOn = False
            for _ in range(P['ITI']): self.tick(z, False)
        if engaged:
            self.emit_slots(out_seq, 5, True)
        self.trials += 1
        if P['turnover'] and self.trials > P['warmupTrials']:
            self.prune_sweep()


def apply_v5_profile(P):
    """THE GRADUATED PHYSICS (2026-08-09, stage B10: canonical trio 15/15). Organism-side
    profile of the value-drive era — every number below was earned by a measured failure and
    verified by an isolated rig (see memory: wanning-neo-value-organ / wanning-neo-valtd-arc).
    Defaults in P stay untouched so every pre-graduation regression stays bit-exact; worlds of
    the new era call this once. World-side economy (KFRUST=1.0 attribution-attenuated x0.5,
    DAYBURN=0.17/day, rScale=meal-proportional teaching, ecological AVAIL) lives in the world."""
    P.update(dict(
        needs=1, valcells=1,
        valTD=1, tdCross=0, valEta=0.15, kCert=3.8, kWallC=0.2,
        valEvid=1, extinctEvid=1, extinctRef=0.5,
        valMemOnly=1, gVal=1.5, thetaUp=0.24, valLam=0.0004,
        usVave=1, vavGov=1,
        allies=1, h0Want=0.10, kBrave=0.6, hunGovW=0.4, vapGovW=4.0,
        kDespV=1.0, kHunSig=0.15, sigmaGoV=0.25,
        curio=1, kCurio=0.25, tauHab=15000.0,
        valSlowE=0,
    ))
    return P

def apply_v6_profile(P):
    """THE CELLIZED PHYSICS (Route A, Lee 2026-08-12): v5 graduation + the adopted
    cellization stack. Every constant below was measured (see CALIBRATION_LEDGER /
    CELLIZATION_METHODS_DRAFT); adoption gates: H channel 30-life exit PASS; always-on
    substrate battery 6-seed full marks; fear trilogy round-2 pack c1 22/c2 17/c3 30/
    c5 25/ALL 13 (conjunction exceeds gated 11; C5 trade documented as phenotype).
    Remaining declared scaffolds: trial reset, position clock, OU noise, substrate
    critic teaching, host novelty bonus (cell chain at 25/30, SI)."""
    apply_v5_profile(P)
    P.update(dict(
        hchan=1,                      # hunger as a ligand (stages 2-4)
        naWire=1, achWire=1,          # arousal + cue-salience by anatomy (debt clearing)
        alwaysOn=1, lamScale=0.0975,  # no teaching boolean; measured value-era rent
        datPump=1,                    # reuptake: tails die by chemistry
        vScale=1,                     # Turrigiano floor: silence is corrected
        matAve=20000.0,               # aversive late maturation (Sullivan)
        fearAnchor=1, reliefCap=1,    # fear consolidates under its own sign; relief pays the no
    ))
    return P

# ---------------- smoke probes (run: python neo_core.py) ----------------
def probe_hold():
    """stage0-style: does plateau memory hold a cue, sparse & graded (cenVar alive)?"""
    print("=== neo port probe 1: plateau HOLD (stage0 pass bars: sparsity 0.05-0.30, cenVar>0.02) ===")
    for d in (0, 8, 16, 30):
        sp = cv = 0.0; S = (3, 7, 11, 23)
        for s in S:
            b = Brain(s, nSym=4, nMem=160)
            b.trial_reset(); b.encode_seq([0])
            z = np.zeros(b.N)
            for _ in range(d): b.tick(z, False)
            a = b.act[b._mem]; c = b.cen[b._mem]
            sp += (a > 0.3).mean(); cv += c.var()
        sp /= len(S); cv /= len(S)
        ok = "PASS" if (0.05 <= sp <= 0.30 and cv > 0.02) else "FAIL"
        print(f"  delay={d:2d}  sparsity={sp:.3f}  cenVar={cv:.4f}   {ok}")

def probe_order():
    """AB vs BA memory states must differ (position binding)."""
    print("=== neo port probe 2: order separation (AB vs BA plat-state cosine < 0.6) ===")
    cs = []
    for s in (3, 7, 11, 23):
        b = Brain(s, nSym=4, nMem=160)
        b.trial_reset(); b.encode_seq([0, 1]); ab = b.plat[b._mem].copy()
        b.trial_reset(); b.encode_seq([1, 0]); ba = b.plat[b._mem].copy()
        cos = float(ab @ ba / (np.linalg.norm(ab) * np.linalg.norm(ba) + 1e-9))
        cs.append(cos)
    m = np.mean(cs)
    print(f"  mean cosine(AB,BA)={m:.3f}  per-seed={[f'{c:.2f}' for c in cs]}   {'PASS' if m < 0.6 else 'FAIL'}")

def probe_learn():
    """The target behaviour: does the port learn 12->345 (single lesson, like the browser tool)?"""
    print("=== neo port probe 3: learn 12->345 (browser-tool equivalent; ~1 min/seed) ===")
    for s in (1, 7):
        b = Brain(s, nSym=8, nMem=160)
        for _ in range(1200): b.train_trial([1, 2], [3, 4, 5])
        said = b.ask([1, 2])
        print(f"  brain #{s}: she says 12 -> {''.join(map(str, said)) or '(silent)'}   {'PASS' if said == [3, 4, 5] else 'FAIL'}")

if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "hold"): probe_hold()
    if which in ("all", "order"): probe_order()
    if which in ("all", "learn"): probe_learn()
