export const profileData = {
  name: 'Duo Zhou',
  intro: `Hi, I'm a Ph.D. student of computer science at the <a href="https://illinois.edu/" target="_blank" rel="noopener">University of Illinois Urbana-Champaign</a> (UIUC). My research interests are in AI Safety, focusing on the theory and application of the following areas:`,
  researchInterests: [
    {
      title: 'Trustworthy Machine Learning',
      description: `I build ML systems that remain reliable far beyond the i.i.d. setting. My work spans distribution shift (<a href="https://ieeexplore.ieee.org/abstract/document/10065450" target="_blank" rel="noopener">AdaNet</a>), adversarial robustness, and <em>certifiable</em> safety via neural network verification-designing algorithms with proofs and scalable implementations (<a href="https://neurips.cc/virtual/2025/poster/118862" target="_blank" rel="noopener">Clip-and-Verify</a>, <a href="https://fmcad.org/FMCAD25/student_forum/" target="_blank" rel="noopener">Lookahead Branching</a>, <a href="https://proceedings.neurips.cc/paper_files/paper/2024/hash/33d93e4dc57453e7667b20f62e7c0681-Abstract-Conference.html" target="_blank" rel="noopener">BICCOS</a>) and studying attack-defense dynamics with formal guarantees.`
    },
    {
      title: 'Decision Making Under Uncertainty',
      description: `I develop reinforcement learning, online learning, game-theoretic methods, and optimal control that optimize worst-case and risk-sensitive performance under uncertainty. Recent projects include robust RL (<a href="https://arxiv.org/pdf/2506.12622" target="_blank" rel="noopener">DR-SAC</a>), multi-agent market simulations under information asymmetry (<a href="https://arxiv.org/pdf/2509.01813" target="_blank" rel="noopener">ShortageSim</a>), and <a href="https://ieeexplore.ieee.org/abstract/document/10644179" target="_blank" rel="noopener">distributionally robust path-integral control</a>.`
    },
    {
      title: 'Optimization & Learning Theory Foundations',
      description: `I use convex/nonconvex optimization, duality, and learning theory to derive finite-sample guarantees and stability-robustness trade-offs, then turn them into practical algorithms (<a href="https://arxiv.org/pdf/2511.17100" target="_blank" rel="noopener">DR-SAC</a>). Current interests include multi-stage, bi-level and semi-infinite formulations for unlearning (<a href="https://arxiv.org/pdf/2506.12622" target="_blank" rel="noopener">GU</a>), certification, distributionally robust objectives, and generalization under strategic or feedback-coupled environments.`
    }
  ],
  links: [
    { label: 'Email: duozhou2 at illinois dot edu', url: 'mailto:duozhou2@illinois.edu' },
    { label: 'Google Scholar', url: 'https://scholar.google.com/citations?user=QnBzRsIAAAAJ&hl=en' },
    { label: 'GitHub', url: 'https://github.com/Lemutisme' }
  ]
};

export const newsData = [
  { date: '[May. 2026]', body: 'InvSTG-PLM accepted to KDD 2026.' },
  { date: '[Apr. 2026]', body: 'Lookahead Branching accepted to IJCAI 2026.' },
  { date: '[Mar. 2026]', body: 'AdaFuse accepted to ACL 2026.' },
  { date: '[Jan. 2026]', body: 'DR-SAC accepted to ICLR 2026.' },
  { date: '[Nov. 2025]', body: 'ShortageSim accepted to AAAI 2026, Oral Presentation.' },
  { date: '[Sept. 2025]', body: 'Clip-and-Verify accepted to NeurIPS 2025.' },
  { date: '[Jul 2025]', body: 'Won both Regular & Extended tracks at VNN-COMP 2025.' },
  { date: '[Sept. 2024]', body: 'BICCOS accepted to NeurIPS 2024.' },
  { date: '[Jul. 2025]', body: 'DR-Path Integral Control presented in ISMP 2024.' },
  { date: '[Jul 2024]', body: 'Won both Regular & Extended tracks at VNN-COMP 2024.' }
];

export const publicationsData = [
  {
    title: 'Agentic Reasoning for Large Language Models',
    authors: 'Tianxin Wei, Ting-Wei Li, Zhining Liu, Xuying Ning, Ze Yang, Jiaru Zou, Zhichen Zeng, Ruizhong Qiu, Xiao Lin, Dongqi Fu, Zihao Li, Mengting Ai, Duo Zhou, Wenxuan Bao, Yunzhe Li, Gaotang Li, Cheng Qian, Yu Wang, Xiangru Tang, Yin Xiao, Liri Fang, Hui Liu, Xianfeng Tang, Yuji Zhang, Chi Wang, Jiaxuan You, Heng Ji, Hanghang Tong, Jingrui He',
    venue: 'preprint, 2026',
    links: [
      { label: 'arXiv', url: 'https://arxiv.org/pdf/2601.12538' },
      { label: 'code', url: 'https://github.com/weitianxin/Awesome-Agentic-Reasoning' }
    ]
  },
  {
    title: 'Geometric-disentanglement Unlearning',
    authors: '<strong>Duo Zhou*</strong>, Yuji Zhang*, Tianxin Wei, Ruizhong Qiu, Ke Yang, Xiao Lin, Cheng Qian, Jingrui He, Hanghang Tong, Heng Ji, Huan Zhang',
    venue: 'preprint, 2025',
    links: [
      { label: 'arXiv', url: 'https://arxiv.org/pdf/2511.17100' },
      { label: 'code', url: 'https://github.com/Lemutisme/Geometric-Unlearning' }
    ]
  },
  {
    title: 'Lookahead Branching for Neural Network Verification',
    authors: 'Liam Davis, <strong>Duo Zhou</strong>, Huan Zhang, Clark Barrett, Guy Katz, Haoze Wu',
    venue: 'Accepted by the 35th International Joint Conference on Artificial Intelligence',
    links: [
      { label: 'IJCAI 2026', url: 'https://openreview.net/forum?id=lX9Eto4L1O' }
    ]
  },
  {
    title: 'DR-SAC: Distributionally Robust Soft Actor-Critic for Reinforcement Learning under Uncertainty',
    authors: 'Mingxuan Cui*, <strong>Duo Zhou*<sup>†</sup></strong>, Yuxuan Han, Grani A Hanasusanto, Qiong Wang, Huan Zhang, Zhengyuan Zhou',
    venue: 'The Fourteenth International Conference on Learning Representations',
    links: [
      { label: 'ICLR 2026', url: 'https://openreview.net/forum?id=a19MA0ksbc&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)' },
      { label: 'code', url: 'https://github.com/Lemutisme/DR-SAC' }
    ]
  },
  {
    title: 'ShortageSim: Simulating Drug Shortages under Information Asymmetry',
    authors: 'Mingxuan Cui*, Yilan Jiang*, <strong>Duo Zhou*</strong>, Cheng Qian, Yuji Zhang<sup>†</sup>, Qiong Wang<sup>†</sup>',
    venue: '(Oral Presentation) Proceedings of the AAAI Conference on Artificial Intelligence, 40(45), 38321-38330. https://doi.org/10.1609/aaai.v40i45.41172',
    links: [
      { label: 'AAAI 2026', url: 'https://ojs.aaai.org/index.php/AAAI/article/view/41172' },
      { label: 'code', url: 'https://github.com/Lemutisme/ShortageSim' }
    ]
  },
  {
    title: 'Clip-and-Verify: Linear Constraint-Driven Domain Clipping for Accelerating Neural Network Verification',
    authors: '<strong>Duo Zhou*</strong>, Jorge Chavez*, Hesun Chen, Grani A. Hanasusanto, Huan Zhang',
    venue: 'Advances in Neural Information Processing Systems, 38, 174849-174895.',
    links: [
      { label: 'NeurIPS 2025', url: 'https://neurips.cc/virtual/2025/poster/118862' },
      { label: 'code', url: 'https://github.com/Verified-Intelligence/Clip_and_Verify' }
    ]
  },
  {
    title: 'Scalable Neural Network Verification with Branch-and-bound Inferred Cutting Planes',
    authors: '<strong>Duo Zhou</strong>, Christopher Brix, Grani A Hanasusanto, Huan Zhang',
    venue: 'Advances in Neural Information Processing Systems, 37, 29324-29353.',
    links: [
      { label: 'NeurIPS 2024', url: 'https://proceedings.neurips.cc/paper_files/paper/2024/hash/33d93e4dc57453e7667b20f62e7c0681-Abstract-Conference.html' },
      { label: 'code', url: 'https://github.com/Lemutisme/BICCOS' }
    ]
  },
  {
    title: 'Distributionally robust path integral control',
    authors: 'Hyuk Park, <strong>Duo Zhou</strong>, Grani A Hanasusanto, Takashi Tanaka',
    venue: 'American Control Conference (ACC), pp. 1164-1171. IEEE, 2024.',
    links: [
      { label: 'ACC 2024', url: 'https://ieeexplore.ieee.org/abstract/document/10644179' },
      { label: 'code', url: 'https://github.com/Lemutisme/Distributionally-Robust-Path-Integral-Control' }
    ]
  },
  {
    title: 'An adaptive and dynamical neural network for machine remaining useful life prediction',
    authors: 'Ruibing Jin*, <strong>Duo Zhou*</strong>, Min Wu, Xiaoli Li, Zhenghua Chen',
    venue: 'IEEE Transactions on Industrial Informatics 20, no. 2 (2023): 1093-1102, 2023',
    links: [
      { label: 'IEEE TII', url: 'https://ieeexplore.ieee.org/abstract/document/10065450' }
    ]
  }
];

export const honorsData = [
  'The Winner for both Regular & Extended Track in VNN-COMP 2025.',
  'The Winner for both Regular & Extended Track in VNN-COMP 2024.',
  'First-Class Scholarship for the Graduating Class. 2021.',
  'Meritorious Winner of Mathematical Contest in Modeling and Interdisciplinary Contest in Modeling. 2020.',
  'Provincial First Prize, Contemporary Undergraduate Mathematical Contest in Modeling. 2019.'
];

export const servicesData = [
  'Reviewer: JMLR, NeurIPS, ICLR, ICML, AAAI, IJCAI, KDD, ARR'
];
