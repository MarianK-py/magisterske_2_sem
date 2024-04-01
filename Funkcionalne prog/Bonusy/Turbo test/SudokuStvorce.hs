module SudokuStvorce where

-- Marian Kravec

sudokuStvorce :: [[Int]] -> [[Int]]
sudokuStvorce sud = [[ sud!!((3*(x `div` 3))+(y `div` 3))!!((3*(x `mod` 3))+(y `mod` 3)) | y <- [0..8] ] | x <- [0..8]]
