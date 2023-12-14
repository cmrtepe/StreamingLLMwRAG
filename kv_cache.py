import torch


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]

    def __call__(self, past_key_values, keeps = None):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        if keeps is None:
            return [
                [
                    torch.cat(
                        [
                            self.k_slice(k, 0, self.start_size),
                            self.k_slice(k, seq_len - self.recent_size, seq_len),
                        ],
                        dim=self.k_seq_dim,
                    ),
                    torch.cat(
                        [
                            self.v_slice(v, 0, self.start_size),
                            self.v_slice(v, seq_len - self.recent_size, seq_len),
                        ],
                        dim=self.v_seq_dim,
                    ),
                ]
                for k, v in past_key_values
            ]
        else:
            keepsize = keeps[0]
            return [
                [
                    torch.cat(
                        [self.k_slice(k, 0, self.start_size)]+
                        [self.k_slice(k, st-1, ed + 1) for st, ed in keeps[1]]+
                        [self.k_slice(k, seq_len - self.recent_size, seq_len)],
                        dim=self.k_seq_dim,
                    ),
                    torch.cat(
                        [self.v_slice(v, 0, self.start_size)]+
                        [self.v_slice(v, st-1, ed + 1) for st, ed in keeps[1]]+
                        [self.v_slice(v, seq_len - self.recent_size, seq_len)],
                        dim=self.v_seq_dim,
                    ),
                ]
                for k, v in past_key_values
            ]

    def evict_for_space(self, past_key_values, num_coming, keeps=None):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        if keeps is None:
            print("evicting...")
            return [
                [
                    torch.cat(
                        [
                            self.k_slice(k, 0, self.start_size),
                            self.k_slice(
                                k, seq_len - self.recent_size + num_coming, seq_len
                            ),
                        ],
                        dim=self.k_seq_dim,
                    ),
                    torch.cat(
                        [
                            self.v_slice(v, 0, self.start_size),
                            self.v_slice(
                                v, seq_len - self.recent_size + num_coming, seq_len
                            ),
                        ],
                        dim=self.v_seq_dim,
                    ),
                ]
                for k, v in past_key_values
            ]
        else:
            keepsize = keeps[0]
            print("evicting...")
            return [
                [
                    torch.cat(
                        [self.k_slice(k, 0, self.start_size)]+
                        [self.k_slice(k, st, ed) for st, ed in keeps[1]]+
                        [self.k_slice(k, seq_len - self.recent_size + num_coming + keepsize//4, seq_len)],
                        dim=self.k_seq_dim,
                    ),
                    torch.cat(
                        [self.v_slice(v, 0, self.start_size)]+
                        [self.v_slice(v, st, ed) for st, ed in keeps[1]]+
                        [self.v_slice(v, seq_len - self.recent_size + num_coming + keepsize//4, seq_len)],
                        dim=self.v_seq_dim,
                    ),
                ]
                for k, v in past_key_values
            ]
            

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
