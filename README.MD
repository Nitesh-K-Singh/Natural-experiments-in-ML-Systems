






                ┌──────────────────────────────┐
                │        True Environment      │
                │  Features X, θ(x), τ(x), ε_t │
                └───────────────┬──────────────┘
                                │
                                │
                        Time Period 0
                                │
                                ▼
                    Historical Dataset
                       (No discount)
                       (X , Y(0))
                                │
                                │
                        Train ML Model
                     Predict churn risk p̂₀(x)
                                │
                                │
                         Deploy Policy
                  Treat top K predicted churn
                                │
                                ▼
                        Time Period 1
                                │
                   Data after intervention
                        (X, D₁, Y₁)
                                │
                                │
                ┌───────────────┼────────────────┐
                │               │                │
                ▼               ▼                ▼
           Fork 1           Fork 2           Fork 3
         Retrain ML        Uplift Model        RDD
      Train P(Y|X,D)      Estimate τ(x)     Estimate τ(x)
     (selection bias)    from treated vs   locally at score
                           untreated          cutoff
                │               │                │
                └───────┬───────┴───────┬────────┘
                        │               │
                        ▼               ▼
                    New Policy      New Policy
                        │               │
                        ▼               ▼
                        Time Period 2
                            │
                     New Outcomes
                       (X, D₂, Y₂)
                            │
                            ▼
                     Repeat Process
                       t = 3 ... 20