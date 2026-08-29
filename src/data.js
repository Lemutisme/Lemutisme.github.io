export const profileData = {
  name: 'Duo Zhou',
  intro: `I am a Computer Science Ph.D. student at the <a href="https://illinois.edu/" target="_blank" rel="noopener">University of Illinois Urbana-Champaign</a>. I work on reliable AI systems: certifying neural networks, making learning robust under uncertainty, and developing principled methods for AI safety.`,
  researchInterests: [
    {
      code: '01',
      title: 'Trustworthy Agentic AI',
      description: `I study reliable agents and multi-agent systems that can plan, coordinate, and improve over long horizons. My current focus includes recursive self-improvement (RSI), agent governance, verifiable execution, and the system foundations needed to keep increasingly capable agents aligned and auditable.`,
    },
    {
      code: '02',
      title: 'Optimization, Post-Training & Verification',
      description: `I develop optimization methods and theory for model post-training, reinforcement learning, and neural network verification. The goal is to connect principled objectives and formal guarantees with scalable algorithms that work on real systems.`,
    },
  ],
  links: [
    { label: 'Email', url: 'mailto:duozhou2@illinois.edu' },
    { label: 'Scholar', url: 'https://scholar.google.com/citations?user=QnBzRsIAAAAJ&hl=en' },
    { label: 'GitHub', url: 'https://github.com/Lemutisme' },
  ],
};

export const newsData = [
  { date: 'Aug. 2026', body: '<strong>Geometric-Disentanglement Language Model Unlearning</strong> accepted to EMNLP 2026 Main.' },
  { date: 'May 2026', body: '<strong>InvSTG-PLM</strong> accepted to KDD 2026.' },
  { date: 'Apr. 2026', body: '<strong>Lookahead Branching</strong> accepted to IJCAI 2026.' },
  { date: 'Mar. 2026', body: '<strong>AdaFuse</strong> accepted to ACL 2026 Main.' },
  { date: 'Jan. 2026', body: '<strong>DR-SAC</strong> accepted to ICLR 2026.' },
  { date: 'Nov. 2025', body: '<strong>ShortageSim</strong> accepted to AAAI 2026 as an oral presentation.' },
];

export const publicationsData = [
  {
    title: 'Geometric-Disentanglement Language Model Unlearning',
    authors: '<strong>Duo Zhou*</strong>, Yuji Zhang*, Tianxin Wei, Ruizhong Qiu, Ke Yang, Xiao Lin, Cheng Qian, Jingrui He, Hanghang Tong, Heng Ji, Huan Zhang',
    venue: 'Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing · Main Conference',
    venueShort: "EMNLP '26",
    year: '2026',
    featured: true,
    links: [
      { label: 'arXiv', url: 'https://arxiv.org/abs/2511.17100' },
      { label: 'Code', url: 'https://github.com/Lemutisme/Geometric-Unlearning' },
    ],
  },
  {
    title: 'Invariant Structure Learning with Pre-trained Language Models for Spatio-temporal Graph',
    authors: 'Ting Wang*, <strong>Duo Zhou*</strong>, Daqian Shi, Hao Tang, Hao Deng, Shengjie Zhao',
    venue: 'Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining',
    venueShort: "KDD '26",
    year: '2026',
    links: [
      { label: 'Paper', url: 'https://doi.org/10.1145/3770855.3818006' },
    ],
  },
  {
    title: 'AdaFuse: Adaptive Ensemble Decoding for Large Language Models',
    authors: 'Chengming Cui*, Tianxin Wei*, Ziyi Chen, Ruizhong Qiu, Zhichen Zeng, Zhining Liu, Xuying Ning, <strong>Duo Zhou</strong>, Jingrui He',
    venue: 'Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics · Main Conference',
    venueShort: "ACL '26",
    year: '2026',
    links: [
      { label: 'Paper', url: 'https://aclanthology.org/2026.acl-long.1974/' },
      { label: 'Code', url: 'https://github.com/CCM0111/AdaFuse' },
    ],
  },
  {
    title: 'Lookahead Branching for Neural Network Verification',
    authors: 'Liam Davis, <strong>Duo Zhou</strong>, Huan Zhang, Guy Katz, Clark W. Barrett, Haoze Wu',
    venue: 'Proceedings of the 35th International Joint Conference on Artificial Intelligence',
    venueShort: "IJCAI '26",
    year: '2026',
    links: [
      { label: 'Paper', url: 'https://openreview.net/forum?id=lX9Eto4L1O' },
    ],
  },
  {
    title: 'DR-SAC: Distributionally Robust Soft Actor-Critic for Reinforcement Learning under Uncertainty',
    authors: 'Mingxuan Cui*, <strong>Duo Zhou*<sup>†</sup></strong>, Yuxuan Han, Grani A. Hanasusanto, Qiong Wang, Huan Zhang, Zhengyuan Zhou',
    venue: 'The Fourteenth International Conference on Learning Representations',
    venueShort: "ICLR '26",
    year: '2026',
    links: [
      { label: 'Paper', url: 'https://openreview.net/forum?id=a19MA0ksbc' },
      { label: 'Code', url: 'https://github.com/Lemutisme/DR-SAC' },
    ],
  },
  {
    title: 'E-Globe: Scalable ε-Global Verification of Neural Networks via Tight Upper Bounds and Pattern-Aware Branching',
    authors: 'Wenting Li, Saif R. Kazi, Russell Bent, <strong>Duo Zhou</strong>, Huan Zhang',
    venue: 'arXiv preprint arXiv:2602.05068',
    venueShort: 'PREPRINT',
    year: '2026',
    links: [
      { label: 'arXiv', url: 'https://arxiv.org/abs/2602.05068' },
    ],
  },
  {
    title: 'Agentic Reasoning for Large Language Models',
    authors: 'Tianxin Wei, Ting-Wei Li, Zhining Liu, Xuying Ning, Ze Yang, Jiaru Zou, Zhichen Zeng, Ruizhong Qiu, Xiao Lin, Dongqi Fu, Zihao Li, Mengting Ai, <strong>Duo Zhou</strong>, Wenxuan Bao, Yunzhe Li, Gaotang Li, Cheng Qian, Yu Wang, Xiangru Tang, Yin Xiao, Liri Fang, Hui Liu, Xianfeng Tang, Yuji Zhang, Chi Wang, Jiaxuan You, Heng Ji, Hanghang Tong, Jingrui He',
    venue: 'arXiv preprint arXiv:2601.12538',
    venueShort: 'SURVEY',
    year: '2026',
    links: [
      { label: 'arXiv', url: 'https://arxiv.org/abs/2601.12538' },
      { label: 'Project', url: 'https://github.com/weitianxin/Awesome-Agentic-Reasoning' },
    ],
  },
  {
    title: 'ShortageSim: Simulating Drug Shortages under Information Asymmetry',
    authors: 'Mingxuan Cui*, Yilan Jiang*, <strong>Duo Zhou*</strong>, Cheng Qian, Yuji Zhang<sup>†</sup>, Qiong Wang<sup>†</sup>',
    venue: 'AAAI 2026 · Oral Presentation',
    venueShort: "AAAI '26",
    year: '2026',
    links: [
      { label: 'Paper', url: 'https://ojs.aaai.org/index.php/AAAI/article/view/41172' },
      { label: 'Code', url: 'https://github.com/Lemutisme/ShortageSim' },
    ],
  },
  {
    title: 'Clip-and-Verify: Linear Constraint-Driven Domain Clipping for Accelerating Neural Network Verification',
    authors: '<strong>Duo Zhou*</strong>, Jorge Chavez*, Hesun Chen, Grani A. Hanasusanto, Huan Zhang',
    venue: 'Advances in Neural Information Processing Systems 38, 174849–174895',
    venueShort: "NEURIPS '25",
    year: '2025',
    links: [
      { label: 'Paper', url: 'https://neurips.cc/virtual/2025/poster/118862' },
      { label: 'Code', url: 'https://github.com/Verified-Intelligence/Clip_and_Verify' },
    ],
  },
  {
    title: 'Scalable Neural Network Verification with Branch-and-bound Inferred Cutting Planes',
    authors: '<strong>Duo Zhou</strong>, Christopher Brix, Grani A. Hanasusanto, Huan Zhang',
    venue: 'Advances in Neural Information Processing Systems 37, 29324–29353',
    venueShort: "NEURIPS '24",
    year: '2024',
    links: [
      { label: 'Paper', url: 'https://proceedings.neurips.cc/paper_files/paper/2024/hash/33d93e4dc57453e7667b20f62e7c0681-Abstract-Conference.html' },
      { label: 'Code', url: 'https://github.com/Lemutisme/BICCOS' },
    ],
  },
  {
    title: 'Distributionally Robust Path Integral Control',
    authors: 'Hyuk Park, <strong>Duo Zhou</strong>, Grani A. Hanasusanto, Takashi Tanaka',
    venue: 'American Control Conference, pp. 1164–1171',
    venueShort: "ACC '24",
    year: '2024',
    links: [
      { label: 'Paper', url: 'https://ieeexplore.ieee.org/abstract/document/10644179' },
      { label: 'Code', url: 'https://github.com/Lemutisme/Distributionally-Robust-Path-Integral-Control' },
    ],
  },
  {
    title: 'An Adaptive and Dynamical Neural Network for Machine Remaining Useful Life Prediction',
    authors: 'Ruibing Jin*, <strong>Duo Zhou*</strong>, Min Wu, Xiaoli Li, Zhenghua Chen',
    venue: 'IEEE Transactions on Industrial Informatics 20(2), 1093–1102',
    venueShort: 'IEEE TII',
    year: '2024',
    links: [
      { label: 'Paper', url: 'https://ieeexplore.ieee.org/abstract/document/10065450' },
    ],
  },
];

export const honorsData = [
  'Winner of both the Regular and Extended tracks at VNN-COMP 2025.',
  'Winner of both the Regular and Extended tracks at VNN-COMP 2024.',
  'First-Class Scholarship for the Graduating Class, 2021.',
  'Meritorious Winner, Mathematical Contest in Modeling and Interdisciplinary Contest in Modeling, 2020.',
  'Provincial First Prize, Contemporary Undergraduate Mathematical Contest in Modeling, 2019.',
];

export const servicesData = [
  'Reviewer: JMLR, NeurIPS, ICLR, ICML, AAAI, IJCAI, KDD, and ARR.',
];
