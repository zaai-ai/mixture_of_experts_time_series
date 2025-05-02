config = {
    "gluonts": {
        "m1_quarterly": {
            "NBEATS": {
                "input_size": 4,
                "mlp_units": [
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ]
                ],
                "n_blocks": [
                    9,
                    9,
                    9
                ],
                "shared_weights": True,
                "scaler_type": "identity",
                "max_steps": 10000,
                "batch_size": 64,
                "windows_batch_size": 256,
                "random_seed": 1,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            },
            "NBeatsMoe": {
                "input_size": 4,
                "mlp_units": [
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ]
                ],
                "n_blocks": [
                    1,
                    1,
                    1
                ],
                "scaler_type": "identity",
                "shared_weights": True,
                "max_steps": 2500,
                "batch_size": 64,
                "windows_batch_size": 1024,
                "random_seed": 3,
                "nr_experts": 8,
                "top_k": 4,
                "early_stop_patience_steps": 10,
                "start_padding_enabled": True
            },
            "nbeatsmoeshared": {
                "input_size": 4,
                "mlp_units": [
                    [
                        8,
                        8
                    ],
                    [
                        8,
                        8
                    ],
                    [
                        8,
                        8
                    ]
                ],
                "n_blocks": [
                    3,
                    3,
                    3
                ],
                "share_experts": True,
                "shared_weights": True,
                "scaler_type": "identity",
                "max_steps": 2500,
                "batch_size": 64,
                "windows_batch_size": 1024,
                "random_seed": 20,
                "nr_experts": 4,
                "top_k": 2,
                "early_stop_patience_steps": 10,
                "start_padding_enabled": True
            },
            "NBeatsStackMoe": {
                "input_size": 4,
                "mlp_units": [
                    [
                        64,
                        64
                    ],
                    [
                        64,
                        64
                    ],
                    [
                        64,
                        64
                    ]
                ],
                "scaler_type": "identity",
                "max_steps": 2500,
                "shared_weights": True,
                "n_blocks": [
                    1,
                    1,
                    1
                ],
                "batch_size": 32,
                "windows_batch_size": 512,
                "random_seed": 18,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            }
        },
        "m1_monthly": {
            "NBEATS": {
                "input_size": 12,
                "mlp_units": [
                    [
                        16,
                        16
                    ],
                    [
                        16,
                        16
                    ],
                    [
                        16,
                        16
                    ]
                ],
                "n_blocks": [
                    9,
                    9,
                    9
                ],
                "shared_weights": True,
                "scaler_type": "identity",
                "max_steps": 2500,
                "batch_size": 32,
                "windows_batch_size": 256,
                "random_seed": 9,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            },
            "NBeatsMoe": {
                "input_size": 12,
                "mlp_units": [
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ]
                ],
                "n_blocks": [
                    3,
                    3,
                    3
                ],
                "scaler_type": "identity",
                "shared_weights": True,
                "max_steps": 2500,
                "batch_size": 32,
                "windows_batch_size": 1024,
                "random_seed": 18,
                "nr_experts": 8,
                "top_k": 2,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            },
            "nbeatsmoeshared": {
                "input_size": 12,
                "mlp_units": [
                    [
                        4,
                        4
                    ],
                    [
                        4,
                        4
                    ],
                    [
                        4,
                        4
                    ]
                ],
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "share_experts": True,
                "shared_weights": True,
                "scaler_type": "identity",
                "max_steps": 5000,
                "batch_size": 32,
                "windows_batch_size": 512,
                "random_seed": 12,
                "nr_experts": 8,
                "top_k": 4,
                "early_stop_patience_steps": 10,
                "start_padding_enabled": True
            },
            "NBeatsStackMoe": {
                "input_size": 12,
                "mlp_units": [
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ]
                ],
                "scaler_type": "identity",
                "max_steps": 1000,
                "shared_weights": True,
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "batch_size": 32,
                "windows_batch_size": 128,
                "random_seed": 4,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            }
        },
        "m1_yearly": {
            "NBEATS": {
                "input_size": 3,
                "mlp_units": [
                    [
                        512,
                        512
                    ],
                    [
                        512,
                        512
                    ],
                    [
                        512,
                        512
                    ]
                ],
                "n_blocks": [
                    9,
                    9,
                    9
                ],
                "shared_weights": True,
                "scaler_type": "identity",
                "max_steps": 10000,
                "batch_size": 32,
                "windows_batch_size": 512,
                "random_seed": 5,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            },
            "NBeatsMoe": {
                "input_size": 3,
                "mlp_units": [
                    [
                        4,
                        4
                    ],
                    [
                        4,
                        4
                    ],
                    [
                        4,
                        4
                    ]
                ],
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "scaler_type": "identity",
                "shared_weights": True,
                "max_steps": 2500,
                "batch_size": 128,
                "windows_batch_size": 128,
                "random_seed": 20,
                "nr_experts": 8,
                "top_k": 2,
                "early_stop_patience_steps": 10,
                "start_padding_enabled": True
            },
            "nbeatsmoeshared": {
                "input_size": 3,
                "mlp_units": [
                    [
                        8,
                        8
                    ],
                    [
                        8,
                        8
                    ],
                    [
                        8,
                        8
                    ]
                ],
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "share_experts": True,
                "shared_weights": True,
                "scaler_type": "identity",
                "max_steps": 10000,
                "batch_size": 128,
                "windows_batch_size": 128,
                "random_seed": 11,
                "nr_experts": 2,
                "top_k": 1,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            },
            "NBeatsStackMoe": {
                "input_size": 3,
                "mlp_units": [
                    [
                        4,
                        4
                    ],
                    [
                        4,
                        4
                    ],
                    [
                        4,
                        4
                    ]
                ],
                "scaler_type": "identity",
                "max_steps": 2500,
                "shared_weights": True,
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "batch_size": 64,
                "windows_batch_size": 512,
                "random_seed": 18,
                "early_stop_patience_steps": 10,
                "start_padding_enabled": True
            }
        },
        "tourism_monthly": {
            "NBEATS": {
                "input_size": 24,
                "mlp_units": [
                    [
                        256,
                        256
                    ],
                    [
                        256,
                        256
                    ],
                    [
                        256,
                        256
                    ]
                ],
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "shared_weights": True,
                "scaler_type": "identity",
                "max_steps": 1000,
                "batch_size": 64,
                "windows_batch_size": 1024,
                "random_seed": 19,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            },
            "NBeatsMoe": {
                "input_size": 24,
                "mlp_units": [
                    [
                        64,
                        64
                    ],
                    [
                        64,
                        64
                    ],
                    [
                        64,
                        64
                    ]
                ],
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "scaler_type": "identity",
                "shared_weights": True,
                "max_steps": 10000,
                "batch_size": 64,
                "windows_batch_size": 1024,
                "random_seed": 7,
                "nr_experts": 4,
                "top_k": 4,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            },
            "nbeatsmoeshared": {
                "input_size": 24,
                "mlp_units": [
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ]
                ],
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "share_experts": True,
                "shared_weights": True,
                "scaler_type": "identity",
                "max_steps": 10000,
                "batch_size": 64,
                "windows_batch_size": 512,
                "random_seed": 20,
                "nr_experts": 8,
                "top_k": 8,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            },
            "NBeatsStackMoe": {
                "input_size": 24,
                "mlp_units": [
                    [
                        1024,
                        1024
                    ],
                    [
                        1024,
                        1024
                    ],
                    [
                        1024,
                        1024
                    ]
                ],
                "scaler_type": "identity",
                "max_steps": 10000,
                "shared_weights": True,
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "batch_size": 128,
                "windows_batch_size": 512,
                "random_seed": 10,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            }
        },
        "tourism_quarterly": {
            "NBEATS": {
                "input_size": 4,
                "mlp_units": [
                    [
                        8,
                        8
                    ],
                    [
                        8,
                        8
                    ],
                    [
                        8,
                        8
                    ]
                ],
                "n_blocks": [
                    9,
                    9,
                    9
                ],
                "shared_weights": True,
                "scaler_type": "identity",
                "max_steps": 5000,
                "batch_size": 32,
                "windows_batch_size": 512,
                "random_seed": 2,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            },
            "NBeatsMoe": {
                "input_size": 4,
                "mlp_units": [
                    [
                        64,
                        64
                    ],
                    [
                        64,
                        64
                    ],
                    [
                        64,
                        64
                    ]
                ],
                "n_blocks": [
                    9,
                    9,
                    9
                ],
                "scaler_type": "identity",
                "shared_weights": True,
                "max_steps": 10000,
                "batch_size": 32,
                "windows_batch_size": 128,
                "random_seed": 3,
                "nr_experts": 4,
                "top_k": 2,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            },
            "nbeatsmoeshared": {
                "input_size": 4,
                "mlp_units": [
                    [
                        64,
                        64
                    ],
                    [
                        64,
                        64
                    ],
                    [
                        64,
                        64
                    ]
                ],
                "n_blocks": [
                    9,
                    9,
                    9
                ],
                "share_experts": True,
                "shared_weights": True,
                "scaler_type": "identity",
                "max_steps": 5000,
                "batch_size": 32,
                "windows_batch_size": 512,
                "random_seed": 11,
                "nr_experts": 8,
                "top_k": 2,
                "early_stop_patience_steps": 10,
                "start_padding_enabled": True
            },
            "NBeatsStackMoe": {
                "input_size": 4,
                "mlp_units": [
                    [
                        64,
                        64
                    ],
                    [
                        64,
                        64
                    ],
                    [
                        64,
                        64
                    ]
                ],
                "scaler_type": "identity",
                "max_steps": 10000,
                "shared_weights": True,
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "batch_size": 256,
                "windows_batch_size": 128,
                "random_seed": 7,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            }
        },
        "tourism_yealy": {
            "NBEATS": {
                "input_size": 3,
                "mlp_units": [
                    [
                        512,
                        512
                    ],
                    [
                        512,
                        512
                    ],
                    [
                        512,
                        512
                    ]
                ],
                "n_blocks": [
                    1,
                    1,
                    1
                ],
                "shared_weights": True,
                "scaler_type": "identity",
                "max_steps": 1000,
                "batch_size": 256,
                "windows_batch_size": 256,
                "random_seed": 11,
                "early_stop_patience_steps": 10,
                "start_padding_enabled": True
            },
            "NBeatsMoe": {
                "input_size": 3,
                "mlp_units": [
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ]
                ],
                "n_blocks": [
                    3,
                    3,
                    3
                ],
                "scaler_type": "identity",
                "shared_weights": True,
                "max_steps": 2500,
                "batch_size": 256,
                "windows_batch_size": 1024,
                "random_seed": 5,
                "nr_experts": 8,
                "top_k": 2,
                "early_stop_patience_steps": 10,
                "start_padding_enabled": True
            },
            "nbeatsmoeshared": {
                "input_size": 3,
                "mlp_units": [
                    [
                        8,
                        8
                    ],
                    [
                        8,
                        8
                    ],
                    [
                        8,
                        8
                    ]
                ],
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "share_experts": True,
                "shared_weights": True,
                "scaler_type": "identity",
                "max_steps": 5000,
                "batch_size": 256,
                "windows_batch_size": 256,
                "random_seed": 14,
                "nr_experts": 8,
                "top_k": 1,
                "early_stop_patience_steps": 10,
                "start_padding_enabled": True
            },
            "NBeatsStackMoe": {
                "input_size": 3,
                "mlp_units": [
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ]
                ],
                "scaler_type": "identity",
                "max_steps": 1000,
                "shared_weights": True,
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "batch_size": 128,
                "windows_batch_size": 1024,
                "random_seed": 8,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            }
        }
    },
    "m3": {
        "Monthly": {
            "NBEATS": {
                "input_size": 36,
                "mlp_units": [
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ]
                ],
                "n_blocks": [
                    9,
                    9,
                    9
                ],
                "shared_weights": True,
                "scaler_type": "identity",
                "max_steps": 2500,
                "batch_size": 128,
                "windows_batch_size": 256,
                "random_seed": 8,
                "early_stop_patience_steps": 10,
                "start_padding_enabled": True
            },
            "NBeatsMoe": {
                "mlp_units": [
                    [
                        32,
                        32
                    ],
                    [
                        32,
                        32
                    ],
                    [
                        32,
                        32
                    ]
                ],
                "n_blocks": [
                    9,
                    9,
                    9
                ],
                "scaler_type": "identity",
                "shared_weights": True,
                "max_steps": 2500,
                "batch_size": 256,
                "windows_batch_size": 512,
                "random_seed": 20,
                "nr_experts": 4,
                "top_k": 4,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True,
                "input_size": 18,
                "step_size": 1
            },
            "nbeatsmoeshared": {
                "mlp_units": [
                    [
                        256,
                        256
                    ],
                    [
                        256,
                        256
                    ],
                    [
                        256,
                        256
                    ]
                ],
                "n_blocks": [
                    1,
                    1,
                    1
                ],
                "scaler_type": "identity",
                "shared_weights": True,
                "max_steps": 1000,
                "batch_size": 256,
                "windows_batch_size": 1024,
                "random_seed": 12,
                "nr_experts": 2,
                "top_k": 2,
                "early_stop_patience_steps": 10,
                "start_padding_enabled": True,
                "input_size": 36,
                "share_experts": True,
                "step_size": 1
            },
            "NBeatsStackMoe": {
                "input_size": 54,
                "mlp_units": [
                    [
                        256,
                        256
                    ],
                    [
                        256,
                        256
                    ],
                    [
                        256,
                        256
                    ]
                ],
                "scaler_type": "identity",
                "max_steps": 1000,
                "shared_weights": True,
                "n_blocks": [
                    9,
                    9,
                    9
                ],
                "batch_size": 256,
                "windows_batch_size": 512,
                "random_seed": 20,
                "early_stop_patience_steps": 10,
                "start_padding_enabled": True
            }
        },
        "Quarterly": {
            "NBEATS": {
                "input_size": 8,
                "mlp_units": [
                    [
                        256,
                        256
                    ],
                    [
                        256,
                        256
                    ],
                    [
                        256,
                        256
                    ]
                ],
                "n_blocks": [
                    3,
                    3,
                    3
                ],
                "shared_weights": True,
                "scaler_type": "identity",
                "max_steps": 1000,
                "batch_size": 32,
                "windows_batch_size": 256,
                "random_seed": 4,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            },
            "NBeatsMoe": {
                "mlp_units": [
                    [
                        4,
                        4
                    ],
                    [
                        4,
                        4
                    ],
                    [
                        4,
                        4
                    ]
                ],
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "scaler_type": "identity",
                "shared_weights": True,
                "max_steps": 5000,
                "batch_size": 128,
                "windows_batch_size": 1024,
                "random_seed": 6,
                "nr_experts": 8,
                "top_k": 8,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True,
                "input_size": 8,
                "step_size": 1
            },
            "nbeatsmoeshared": {
                "mlp_units": [
                    [
                        32,
                        32
                    ],
                    [
                        32,
                        32
                    ],
                    [
                        32,
                        32
                    ]
                ],
                "n_blocks": [
                    3,
                    3,
                    3
                ],
                "scaler_type": "identity",
                "shared_weights": True,
                "max_steps": 10000,
                "batch_size": 32,
                "windows_batch_size": 512,
                "random_seed": 7,
                "nr_experts": 2,
                "top_k": 1,
                "early_stop_patience_steps": 10,
                "start_padding_enabled": True,
                "input_size": 16,
                "share_experts": True,
                "step_size": 1
            },
            "NBeatsStackMoe": {
                "input_size": 8,
                "mlp_units": [
                    [
                        32,
                        32
                    ],
                    [
                        32,
                        32
                    ],
                    [
                        32,
                        32
                    ]
                ],
                "scaler_type": "identity",
                "max_steps": 2500,
                "shared_weights": True,
                "n_blocks": [
                    9,
                    9,
                    9
                ],
                "batch_size": 128,
                "windows_batch_size": 1024,
                "random_seed": 12,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            }
        },
        "Yearly": {
            "NBEATS": {
                "input_size": 6,
                "mlp_units": [
                    [
                        8,
                        8
                    ],
                    [
                        8,
                        8
                    ],
                    [
                        8,
                        8
                    ]
                ],
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "shared_weights": True,
                "scaler_type": "identity",
                "max_steps": 5000,
                "batch_size": 256,
                "windows_batch_size": 256,
                "random_seed": 7,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            },
            "NBeatsMoe": {
                "mlp_units": [
                    [
                        4,
                        4
                    ],
                    [
                        4,
                        4
                    ],
                    [
                        4,
                        4
                    ]
                ],
                "n_blocks": [
                    1,
                    1,
                    1
                ],
                "scaler_type": "identity",
                "shared_weights": True,
                "max_steps": 1000,
                "batch_size": 128,
                "windows_batch_size": 512,
                "random_seed": 9,
                "nr_experts": 8,
                "top_k": 1,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True,
                "input_size": 6,
                "step_size": 1
            },
            "nbeatsmoeshared": {
                "mlp_units": [
                    [
                        8,
                        8
                    ],
                    [
                        8,
                        8
                    ],
                    [
                        8,
                        8
                    ]
                ],
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "scaler_type": "identity",
                "shared_weights": True,
                "max_steps": 2500,
                "batch_size": 128,
                "windows_batch_size": 128,
                "random_seed": 16,
                "nr_experts": 8,
                "top_k": 2,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True,
                "input_size": 6,
                "share_experts": True,
                "step_size": 1
            },
            "NBeatsStackMoe": {
                "input_size": 6,
                "mlp_units": [
                    [
                        16,
                        16
                    ],
                    [
                        16,
                        16
                    ],
                    [
                        16,
                        16
                    ]
                ],
                "scaler_type": "identity",
                "max_steps": 10000,
                "shared_weights": True,
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "batch_size": 32,
                "windows_batch_size": 1024,
                "random_seed": 10,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            }
        }
    },
    "m4": {
        "Monthly": {
            "NBEATS": {
                "input_size": 36,
                "mlp_units": [
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ]
                ],
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "shared_weights": True,
                "scaler_type": "identity",
                "max_steps": 5000,
                "batch_size": 64,
                "windows_batch_size": 512,
                "random_seed": 13,
                "early_stop_patience_steps": 10,
                "start_padding_enabled": True
            },
            "NBeatsMoe": {
                "mlp_units": [
                    [
                        32,
                        32
                    ],
                    [
                        32,
                        32
                    ],
                    [
                        32,
                        32
                    ]
                ],
                "n_blocks": [
                    1,
                    1,
                    1
                ],
                "scaler_type": "identity",
                "shared_weights": True,
                "max_steps": 10000,
                "batch_size": 256,
                "windows_batch_size": 512,
                "random_seed": 20,
                "nr_experts": 8,
                "top_k": 2,
                "early_stop_patience_steps": 10,
                "start_padding_enabled": True,
                "input_size": 54,
                "step_size": 1
            },
            "nbeatsmoeshared": {
                "mlp_units": [
                    [
                        64,
                        64
                    ],
                    [
                        64,
                        64
                    ],
                    [
                        64,
                        64
                    ]
                ],
                "n_blocks": [
                    1,
                    1,
                    1
                ],
                "scaler_type": "identity",
                "shared_weights": True,
                "max_steps": 5000,
                "batch_size": 32,
                "windows_batch_size": 256,
                "random_seed": 20,
                "nr_experts": 8,
                "top_k": 8,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True,
                "input_size": 36,
                "share_experts": True,
                "step_size": 18
            },
            "NBeatsStackMoe": {
                "input_size": 54,
                "mlp_units": [
                    [
                        1024,
                        1024
                    ],
                    [
                        1024,
                        1024
                    ],
                    [
                        1024,
                        1024
                    ]
                ],
                "scaler_type": "identity",
                "max_steps": 10000,
                "shared_weights": True,
                "n_blocks": [
                    9,
                    9,
                    9
                ],
                "batch_size": 32,
                "windows_batch_size": 1024,
                "random_seed": 13,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            }
        },
        "Quarterly": {
            "NBEATS": {
                "input_size": 40,
                "mlp_units": [
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ],
                    [
                        128,
                        128
                    ]
                ],
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "shared_weights": True,
                "scaler_type": "identity",
                "max_steps": 2500,
                "batch_size": 32,
                "windows_batch_size": 512,
                "random_seed": 20,
                "early_stop_patience_steps": 10,
                "start_padding_enabled": True
            },
            "NBeatsMoe": {
                "mlp_units": [
                    [
                        64,
                        64
                    ],
                    [
                        64,
                        64
                    ],
                    [
                        64,
                        64
                    ]
                ],
                "n_blocks": [
                    9,
                    9,
                    9
                ],
                "scaler_type": "identity",
                "shared_weights": True,
                "max_steps": 10000,
                "batch_size": 128,
                "windows_batch_size": 128,
                "random_seed": 14,
                "nr_experts": 8,
                "top_k": 8,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True,
                "input_size": 16,
                "step_size": 1
            },
            "nbeatsmoeshared": {
                "mlp_units": [
                    [
                        512,
                        512
                    ],
                    [
                        512,
                        512
                    ],
                    [
                        512,
                        512
                    ]
                ],
                "n_blocks": [
                    3,
                    3,
                    3
                ],
                "scaler_type": "identity",
                "shared_weights": True,
                "max_steps": 10000,
                "batch_size": 128,
                "windows_batch_size": 512,
                "random_seed": 14,
                "nr_experts": 8,
                "top_k": 1,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True,
                "input_size": 32,
                "share_experts": True,
                "step_size": 1
            },
            "NBeatsStackMoe": {
                "input_size": 16,
                "mlp_units": [
                    [
                        256,
                        256
                    ],
                    [
                        256,
                        256
                    ],
                    [
                        256,
                        256
                    ]
                ],
                "scaler_type": "identity",
                "max_steps": 5000,
                "shared_weights": True,
                "n_blocks": [
                    9,
                    9,
                    9
                ],
                "batch_size": 64,
                "windows_batch_size": 512,
                "random_seed": 13,
                "early_stop_patience_steps": 10,
                "start_padding_enabled": True
            }
        },
        "Yearly": {
            "NBEATS": {
                "input_size": 18,
                "mlp_units": [
                    [
                        512,
                        512
                    ],
                    [
                        512,
                        512
                    ],
                    [
                        512,
                        512
                    ]
                ],
                "n_blocks": [
                    6,
                    6,
                    6
                ],
                "shared_weights": True,
                "scaler_type": "identity",
                "max_steps": 5000,
                "batch_size": 256,
                "windows_batch_size": 512,
                "random_seed": 10,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True
            },
            "NBeatsMoe": {
                "mlp_units": [
                    [
                        8,
                        8
                    ],
                    [
                        8,
                        8
                    ],
                    [
                        8,
                        8
                    ]
                ],
                "n_blocks": [
                    1,
                    1,
                    1
                ],
                "scaler_type": "identity",
                "shared_weights": True,
                "max_steps": 10000,
                "batch_size": 128,
                "windows_batch_size": 128,
                "random_seed": 9,
                "nr_experts": 2,
                "top_k": 2,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True,
                "input_size": 12,
                "step_size": 1
            },
            "nbeatsmoeshared": {
                "mlp_units": [
                    [
                        256,
                        256
                    ],
                    [
                        256,
                        256
                    ],
                    [
                        256,
                        256
                    ]
                ],
                "n_blocks": [
                    3,
                    3,
                    3
                ],
                "scaler_type": "identity",
                "shared_weights": True,
                "max_steps": 2500,
                "batch_size": 256,
                "windows_batch_size": 128,
                "random_seed": 17,
                "nr_experts": 8,
                "top_k": 2,
                "early_stop_patience_steps": 20,
                "start_padding_enabled": True,
                "input_size": 30,
                "share_experts": True,
                "step_size": 1
            },
            "NBeatsStackMoe": {
                "input_size": 18,
                "mlp_units": [
                    [
                        512,
                        512
                    ],
                    [
                        512,
                        512
                    ],
                    [
                        512,
                        512
                    ]
                ],
                "scaler_type": "identity",
                "max_steps": 5000,
                "shared_weights": True,
                "n_blocks": [
                    1,
                    1,
                    1
                ],
                "batch_size": 128,
                "windows_batch_size": 512,
                "random_seed": 15,
                "early_stop_patience_steps": 10,
                "start_padding_enabled": True
            }
        }
    }
}