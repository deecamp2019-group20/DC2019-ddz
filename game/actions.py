# -*- coding: utf-8 -*-
"""
所有的合理出牌方式
卡片数组:编号   
429种  
"""

action_dict = {
     '[1, 1, 1, 10, 10]': 205,
     '[1, 1, 1, 10]': 49,
     '[1, 1, 1, 11, 11]': 206,
     '[1, 1, 1, 11]': 50,
     '[1, 1, 1, 12, 12]': 207,
     '[1, 1, 1, 12]': 51,
     '[1, 1, 1, 13, 13]': 208,
     '[1, 1, 1, 13]': 52,
     '[1, 1, 1, 14]': 416,
     '[1, 1, 1, 15]': 403,
     '[1, 1, 1, 1]': 353,
     '[1, 1, 1, 2, 2]': 197,
     '[1, 1, 1, 2]': 41,
     '[1, 1, 1, 3, 3]': 198,
     '[1, 1, 1, 3]': 42,
     '[1, 1, 1, 4, 4]': 199,
     '[1, 1, 1, 4]': 43,
     '[1, 1, 1, 5, 5]': 200,
     '[1, 1, 1, 5]': 44,
     '[1, 1, 1, 6, 6]': 201,
     '[1, 1, 1, 6]': 45,
     '[1, 1, 1, 7, 7]': 202,
     '[1, 1, 1, 7]': 46,
     '[1, 1, 1, 8, 8]': 203,
     '[1, 1, 1, 8]': 47,
     '[1, 1, 1, 9, 9]': 204,
     '[1, 1, 1, 9]': 48,
     '[1, 1, 10, 10, 10]': 305,
     '[1, 1, 11, 11, 11]': 317,
     '[1, 1, 12, 12, 12]': 329,
     '[1, 1, 13, 13, 13]': 341,
     '[1, 1, 1]': 28,
     '[1, 1, 2, 2, 2]': 209,
     '[1, 1, 3, 3, 3]': 221,
     '[1, 1, 4, 4, 4]': 233,
     '[1, 1, 5, 5, 5]': 245,
     '[1, 1, 6, 6, 6]': 257,
     '[1, 1, 7, 7, 7]': 269,
     '[1, 1, 8, 8, 8]': 281,
     '[1, 1, 9, 9, 9]': 293,
     '[1, 10, 10, 10]': 149,
     '[1, 10, 11, 12, 13]': 374,
     '[1, 11, 11, 11]': 161,
     '[1, 12, 12, 12]': 173,
     '[1, 13, 13, 13]': 185,
     '[1, 1]': 15,
     '[1, 2, 2, 2]': 53,
     '[1, 3, 3, 3]': 65,
     '[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]': 402,
     '[1, 4, 4, 4]': 77,
     '[1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]': 401,
     '[1, 5, 5, 5]': 89,
     '[1, 5, 6, 7, 8, 9, 10, 11, 12, 13]': 399,
     '[1, 6, 6, 6]': 101,
     '[1, 6, 7, 8, 9, 10, 11, 12, 13]': 396,
     '[1, 7, 7, 7]': 113,
     '[1, 7, 8, 9, 10, 11, 12, 13]': 392,
     '[1, 8, 8, 8]': 125,
     '[1, 8, 9, 10, 11, 12, 13]': 387,
     '[1, 9, 10, 11, 12, 13]': 381,
     '[1, 9, 9, 9]': 137,
     '[10, 10, 10, 10]': 362,
     '[10, 10, 10, 11, 11]': 314,
     '[10, 10, 10, 11]': 158,
     '[10, 10, 10, 12, 12]': 315,
     '[10, 10, 10, 12]': 159,
     '[10, 10, 10, 13, 13]': 316,
     '[10, 10, 10, 13]': 160,
     '[10, 10, 10, 14]': 425,
     '[10, 10, 10, 15]': 412,
     '[10, 10, 10]': 37,
     '[10, 10, 11, 11, 11]': 326,
     '[10, 10, 12, 12, 12]': 338,
     '[10, 10, 13, 13, 13]': 350,
     '[10, 10]': 24,
     '[10, 11, 11, 11]': 170,
     '[10, 12, 12, 12]': 182,
     '[10, 13, 13, 13]': 194,
     '[10]': 9,
     '[11, 11, 11, 11]': 363,
     '[11, 11, 11, 12, 12]': 327,
     '[11, 11, 11, 12]': 171,
     '[11, 11, 11, 13, 13]': 328,
     '[11, 11, 11, 13]': 172,
     '[11, 11, 11, 14]': 426,
     '[11, 11, 11, 15]': 413,
     '[11, 11, 11]': 38,
     '[11, 11, 12, 12, 12]': 339,
     '[11, 11, 13, 13, 13]': 351,
     '[11, 11]': 25,
     '[11, 12, 12, 12]': 183,
     '[11, 13, 13, 13]': 195,
     '[11]': 10,
     '[12, 12, 12, 12]': 364,
     '[12, 12, 12, 13, 13]': 340,
     '[12, 12, 12, 13]': 184,
     '[12, 12, 12, 14]': 427,
     '[12, 12, 12, 15]': 414,
     '[12, 12, 12]': 39,
     '[12, 12, 13, 13, 13]': 352,
     '[12, 12]': 26,
     '[12, 13, 13, 13]': 196,
     '[12]': 11,
     '[13, 13, 13, 13]': 365,
     '[13, 13, 13, 14]': 428,
     '[13, 13, 13, 15]': 415,
     '[13, 13, 13]': 40,
     '[13, 13]': 27,
     '[13]': 12,
     '[14, 15]': 366,
     '[14]': 13,
     '[15]': 14,
     '[1]': 0,
     '[2, 10, 10, 10]': 150,
     '[2, 11, 11, 11]': 162,
     '[2, 12, 12, 12]': 174,
     '[2, 13, 13, 13]': 186,
     '[2, 2, 10, 10, 10]': 306,
     '[2, 2, 11, 11, 11]': 318,
     '[2, 2, 12, 12, 12]': 330,
     '[2, 2, 13, 13, 13]': 342,
     '[2, 2, 2, 10, 10]': 217,
     '[2, 2, 2, 10]': 61,
     '[2, 2, 2, 11, 11]': 218,
     '[2, 2, 2, 11]': 62,
     '[2, 2, 2, 12, 12]': 219,
     '[2, 2, 2, 12]': 63,
     '[2, 2, 2, 13, 13]': 220,
     '[2, 2, 2, 13]': 64,
     '[2, 2, 2, 14]': 417,
     '[2, 2, 2, 15]': 404,
     '[2, 2, 2, 2]': 354,
     '[2, 2, 2, 3, 3]': 210,
     '[2, 2, 2, 3]': 54,
     '[2, 2, 2, 4, 4]': 211,
     '[2, 2, 2, 4]': 55,
     '[2, 2, 2, 5, 5]': 212,
     '[2, 2, 2, 5]': 56,
     '[2, 2, 2, 6, 6]': 213,
     '[2, 2, 2, 6]': 57,
     '[2, 2, 2, 7, 7]': 214,
     '[2, 2, 2, 7]': 58,
     '[2, 2, 2, 8, 8]': 215,
     '[2, 2, 2, 8]': 59,
     '[2, 2, 2, 9, 9]': 216,
     '[2, 2, 2, 9]': 60,
     '[2, 2, 2]': 29,
     '[2, 2, 3, 3, 3]': 222,
     '[2, 2, 4, 4, 4]': 234,
     '[2, 2, 5, 5, 5]': 246,
     '[2, 2, 6, 6, 6]': 258,
     '[2, 2, 7, 7, 7]': 270,
     '[2, 2, 8, 8, 8]': 282,
     '[2, 2, 9, 9, 9]': 294,
     '[2, 2]': 16,
     '[2, 3, 3, 3]': 66,
     '[2, 4, 4, 4]': 78,
     '[2, 5, 5, 5]': 90,
     '[2, 6, 6, 6]': 102,
     '[2, 7, 7, 7]': 114,
     '[2, 8, 8, 8]': 126,
     '[2, 9, 9, 9]': 138,
     '[2]': 1,
     '[3, 10, 10, 10]': 151,
     '[3, 11, 11, 11]': 163,
     '[3, 12, 12, 12]': 175,
     '[3, 13, 13, 13]': 187,
     '[3, 3, 10, 10, 10]': 307,
     '[3, 3, 11, 11, 11]': 319,
     '[3, 3, 12, 12, 12]': 331,
     '[3, 3, 13, 13, 13]': 343,
     '[3, 3, 3, 10, 10]': 229,
     '[3, 3, 3, 10]': 73,
     '[3, 3, 3, 11, 11]': 230,
     '[3, 3, 3, 11]': 74,
     '[3, 3, 3, 12, 12]': 231,
     '[3, 3, 3, 12]': 75,
     '[3, 3, 3, 13, 13]': 232,
     '[3, 3, 3, 13]': 76,
     '[3, 3, 3, 14]': 418,
     '[3, 3, 3, 15]': 405,
     '[3, 3, 3, 3]': 355,
     '[3, 3, 3, 4, 4]': 223,
     '[3, 3, 3, 4]': 67,
     '[3, 3, 3, 5, 5]': 224,
     '[3, 3, 3, 5]': 68,
     '[3, 3, 3, 6, 6]': 225,
     '[3, 3, 3, 6]': 69,
     '[3, 3, 3, 7, 7]': 226,
     '[3, 3, 3, 7]': 70,
     '[3, 3, 3, 8, 8]': 227,
     '[3, 3, 3, 8]': 71,
     '[3, 3, 3, 9, 9]': 228,
     '[3, 3, 3, 9]': 72,
     '[3, 3, 3]': 30,
     '[3, 3, 4, 4, 4]': 235,
     '[3, 3, 5, 5, 5]': 247,
     '[3, 3, 6, 6, 6]': 259,
     '[3, 3, 7, 7, 7]': 271,
     '[3, 3, 8, 8, 8]': 283,
     '[3, 3, 9, 9, 9]': 295,
     '[3, 3]': 17,
     '[3, 4, 4, 4]': 79,
     '[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]': 400,
     '[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]': 397,
     '[3, 4, 5, 6, 7, 8, 9, 10, 11]': 393,
     '[3, 4, 5, 6, 7, 8, 9, 10]': 388,
     '[3, 4, 5, 6, 7, 8, 9]': 382,
     '[3, 4, 5, 6, 7, 8]': 375,
     '[3, 4, 5, 6, 7]': 367,
     '[3, 5, 5, 5]': 91,
     '[3, 6, 6, 6]': 103,
     '[3, 7, 7, 7]': 115,
     '[3, 8, 8, 8]': 127,
     '[3, 9, 9, 9]': 139,
     '[3]': 2,
     '[4, 10, 10, 10]': 152,
     '[4, 11, 11, 11]': 164,
     '[4, 12, 12, 12]': 176,
     '[4, 13, 13, 13]': 188,
     '[4, 4, 10, 10, 10]': 308,
     '[4, 4, 11, 11, 11]': 320,
     '[4, 4, 12, 12, 12]': 332,
     '[4, 4, 13, 13, 13]': 344,
     '[4, 4, 4, 10, 10]': 241,
     '[4, 4, 4, 10]': 85,
     '[4, 4, 4, 11, 11]': 242,
     '[4, 4, 4, 11]': 86,
     '[4, 4, 4, 12, 12]': 243,
     '[4, 4, 4, 12]': 87,
     '[4, 4, 4, 13, 13]': 244,
     '[4, 4, 4, 13]': 88,
     '[4, 4, 4, 14]': 419,
     '[4, 4, 4, 15]': 406,
     '[4, 4, 4, 4]': 356,
     '[4, 4, 4, 5, 5]': 236,
     '[4, 4, 4, 5]': 80,
     '[4, 4, 4, 6, 6]': 237,
     '[4, 4, 4, 6]': 81,
     '[4, 4, 4, 7, 7]': 238,
     '[4, 4, 4, 7]': 82,
     '[4, 4, 4, 8, 8]': 239,
     '[4, 4, 4, 8]': 83,
     '[4, 4, 4, 9, 9]': 240,
     '[4, 4, 4, 9]': 84,
     '[4, 4, 4]': 31,
     '[4, 4, 5, 5, 5]': 248,
     '[4, 4, 6, 6, 6]': 260,
     '[4, 4, 7, 7, 7]': 272,
     '[4, 4, 8, 8, 8]': 284,
     '[4, 4, 9, 9, 9]': 296,
     '[4, 4]': 18,
     '[4, 5, 5, 5]': 92,
     '[4, 5, 6, 7, 8, 9, 10, 11, 12, 13]': 398,
     '[4, 5, 6, 7, 8, 9, 10, 11, 12]': 394,
     '[4, 5, 6, 7, 8, 9, 10, 11]': 389,
     '[4, 5, 6, 7, 8, 9, 10]': 383,
     '[4, 5, 6, 7, 8, 9]': 376,
     '[4, 5, 6, 7, 8]': 368,
     '[4, 6, 6, 6]': 104,
     '[4, 7, 7, 7]': 116,
     '[4, 8, 8, 8]': 128,
     '[4, 9, 9, 9]': 140,
     '[4]': 3,
     '[5, 10, 10, 10]': 153,
     '[5, 11, 11, 11]': 165,
     '[5, 12, 12, 12]': 177,
     '[5, 13, 13, 13]': 189,
     '[5, 5, 10, 10, 10]': 309,
     '[5, 5, 11, 11, 11]': 321,
     '[5, 5, 12, 12, 12]': 333,
     '[5, 5, 13, 13, 13]': 345,
     '[5, 5, 5, 10, 10]': 253,
     '[5, 5, 5, 10]': 97,
     '[5, 5, 5, 11, 11]': 254,
     '[5, 5, 5, 11]': 98,
     '[5, 5, 5, 12, 12]': 255,
     '[5, 5, 5, 12]': 99,
     '[5, 5, 5, 13, 13]': 256,
     '[5, 5, 5, 13]': 100,
     '[5, 5, 5, 14]': 420,
     '[5, 5, 5, 15]': 407,
     '[5, 5, 5, 5]': 357,
     '[5, 5, 5, 6, 6]': 249,
     '[5, 5, 5, 6]': 93,
     '[5, 5, 5, 7, 7]': 250,
     '[5, 5, 5, 7]': 94,
     '[5, 5, 5, 8, 8]': 251,
     '[5, 5, 5, 8]': 95,
     '[5, 5, 5, 9, 9]': 252,
     '[5, 5, 5, 9]': 96,
     '[5, 5, 5]': 32,
     '[5, 5, 6, 6, 6]': 261,
     '[5, 5, 7, 7, 7]': 273,
     '[5, 5, 8, 8, 8]': 285,
     '[5, 5, 9, 9, 9]': 297,
     '[5, 5]': 19,
     '[5, 6, 6, 6]': 105,
     '[5, 6, 7, 8, 9, 10, 11, 12, 13]': 395,
     '[5, 6, 7, 8, 9, 10, 11, 12]': 390,
     '[5, 6, 7, 8, 9, 10, 11]': 384,
     '[5, 6, 7, 8, 9, 10]': 377,
     '[5, 6, 7, 8, 9]': 369,
     '[5, 7, 7, 7]': 117,
     '[5, 8, 8, 8]': 129,
     '[5, 9, 9, 9]': 141,
     '[5]': 4,
     '[6, 10, 10, 10]': 154,
     '[6, 11, 11, 11]': 166,
     '[6, 12, 12, 12]': 178,
     '[6, 13, 13, 13]': 190,
     '[6, 6, 10, 10, 10]': 310,
     '[6, 6, 11, 11, 11]': 322,
     '[6, 6, 12, 12, 12]': 334,
     '[6, 6, 13, 13, 13]': 346,
     '[6, 6, 6, 10, 10]': 265,
     '[6, 6, 6, 10]': 109,
     '[6, 6, 6, 11, 11]': 266,
     '[6, 6, 6, 11]': 110,
     '[6, 6, 6, 12, 12]': 267,
     '[6, 6, 6, 12]': 111,
     '[6, 6, 6, 13, 13]': 268,
     '[6, 6, 6, 13]': 112,
     '[6, 6, 6, 14]': 421,
     '[6, 6, 6, 15]': 408,
     '[6, 6, 6, 6]': 358,
     '[6, 6, 6, 7, 7]': 262,
     '[6, 6, 6, 7]': 106,
     '[6, 6, 6, 8, 8]': 263,
     '[6, 6, 6, 8]': 107,
     '[6, 6, 6, 9, 9]': 264,
     '[6, 6, 6, 9]': 108,
     '[6, 6, 6]': 33,
     '[6, 6, 7, 7, 7]': 274,
     '[6, 6, 8, 8, 8]': 286,
     '[6, 6, 9, 9, 9]': 298,
     '[6, 6]': 20,
     '[6, 7, 7, 7]': 118,
     '[6, 7, 8, 9, 10, 11, 12, 13]': 391,
     '[6, 7, 8, 9, 10, 11, 12]': 385,
     '[6, 7, 8, 9, 10, 11]': 378,
     '[6, 7, 8, 9, 10]': 370,
     '[6, 8, 8, 8]': 130,
     '[6, 9, 9, 9]': 142,
     '[6]': 5,
     '[7, 10, 10, 10]': 155,
     '[7, 11, 11, 11]': 167,
     '[7, 12, 12, 12]': 179,
     '[7, 13, 13, 13]': 191,
     '[7, 7, 10, 10, 10]': 311,
     '[7, 7, 11, 11, 11]': 323,
     '[7, 7, 12, 12, 12]': 335,
     '[7, 7, 13, 13, 13]': 347,
     '[7, 7, 7, 10, 10]': 277,
     '[7, 7, 7, 10]': 121,
     '[7, 7, 7, 11, 11]': 278,
     '[7, 7, 7, 11]': 122,
     '[7, 7, 7, 12, 12]': 279,
     '[7, 7, 7, 12]': 123,
     '[7, 7, 7, 13, 13]': 280,
     '[7, 7, 7, 13]': 124,
     '[7, 7, 7, 14]': 422,
     '[7, 7, 7, 15]': 409,
     '[7, 7, 7, 7]': 359,
     '[7, 7, 7, 8, 8]': 275,
     '[7, 7, 7, 8]': 119,
     '[7, 7, 7, 9, 9]': 276,
     '[7, 7, 7, 9]': 120,
     '[7, 7, 7]': 34,
     '[7, 7, 8, 8, 8]': 287,
     '[7, 7, 9, 9, 9]': 299,
     '[7, 7]': 21,
     '[7, 8, 8, 8]': 131,
     '[7, 8, 9, 10, 11, 12, 13]': 386,
     '[7, 8, 9, 10, 11, 12]': 379,
     '[7, 8, 9, 10, 11]': 371,
     '[7, 9, 9, 9]': 143,
     '[7]': 6,
     '[8, 10, 10, 10]': 156,
     '[8, 11, 11, 11]': 168,
     '[8, 12, 12, 12]': 180,
     '[8, 13, 13, 13]': 192,
     '[8, 8, 10, 10, 10]': 312,
     '[8, 8, 11, 11, 11]': 324,
     '[8, 8, 12, 12, 12]': 336,
     '[8, 8, 13, 13, 13]': 348,
     '[8, 8, 8, 10, 10]': 289,
     '[8, 8, 8, 10]': 133,
     '[8, 8, 8, 11, 11]': 290,
     '[8, 8, 8, 11]': 134,
     '[8, 8, 8, 12, 12]': 291,
     '[8, 8, 8, 12]': 135,
     '[8, 8, 8, 13, 13]': 292,
     '[8, 8, 8, 13]': 136,
     '[8, 8, 8, 14]': 423,
     '[8, 8, 8, 15]': 410,
     '[8, 8, 8, 8]': 360,
     '[8, 8, 8, 9, 9]': 288,
     '[8, 8, 8, 9]': 132,
     '[8, 8, 8]': 35,
     '[8, 8, 9, 9, 9]': 300,
     '[8, 8]': 22,
     '[8, 9, 10, 11, 12, 13]': 380,
     '[8, 9, 10, 11, 12]': 372,
     '[8, 9, 9, 9]': 144,
     '[8]': 7,
     '[9, 10, 10, 10]': 157,
     '[9, 10, 11, 12, 13]': 373,
     '[9, 11, 11, 11]': 169,
     '[9, 12, 12, 12]': 181,
     '[9, 13, 13, 13]': 193,
     '[9, 9, 10, 10, 10]': 313,
     '[9, 9, 11, 11, 11]': 325,
     '[9, 9, 12, 12, 12]': 337,
     '[9, 9, 13, 13, 13]': 349,
     '[9, 9, 9, 10, 10]': 301,
     '[9, 9, 9, 10]': 145,
     '[9, 9, 9, 11, 11]': 302,
     '[9, 9, 9, 11]': 146,
     '[9, 9, 9, 12, 12]': 303,
     '[9, 9, 9, 12]': 147,
     '[9, 9, 9, 13, 13]': 304,
     '[9, 9, 9, 13]': 148,
     '[9, 9, 9, 14]': 424,
     '[9, 9, 9, 15]': 411,
     '[9, 9, 9, 9]': 361,
     '[9, 9, 9]': 36,
     '[9, 9]': 23,
     '[9]': 8
}