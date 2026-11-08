# Interview defence

**How did you prevent temporal leakage?** “I split complete process runs, fitted preprocessing on training runs only, and tested prefix invariance: changing future observations cannot change a forecast already made.” Здесь речь об online information set, а не об идентификации причинных эффектов.

**Why model error correlation?** “Physics and neural forecasts share inputs and therefore correlated errors. I estimated the cross-error second moment on validation data and compared it with an independence assumption.” Выведите формулу веса из производной квадратичной ошибки смеси.

**What evidence supports improvement?** “On a controlled simulator, three training seeds gave lower mean observed-target RMSE for fusion than the mechanistic baseline at the reported horizons. I retained per-run predictions and used whole-run bootstrap intervals. This does not establish real-plant transfer.” Отдельно назовите восемь test-прогонов и наличие перекрывающихся окон.

**What would you do with more compute?** “I would prioritise structurally different simulated regimes and licensed plant data, then evaluate temporal architectures under the same causal replay protocol.” Увеличение GRU само по себе не устраняет bias симулятора.

**What was implemented?** Реально исполнены симулятор, causal replay, masked GRU training, checkpoint reload, two fusion baselines, test/OOD evaluation и тесты. Код создан с помощью AI coding assistant. Самостоятельная защита требует воспроизведения и понимания; не утверждайте единоличное ручное авторство или промышленное внедрение.
