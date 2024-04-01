module SudokuStvorce where
import Data.List

-- Marian Kravec

-- moje riesenie z Turbo test
sudokuStvorce :: [[Int]] -> [[Int]]
sudokuStvorce sud = [[ sud!!((3*(x `div` 3))+(y `div` 3))!!((3*(x `mod` 3))+(y `mod` 3)) | y <- [0..8] ] | x <- [0..8]]

skontrRiadok :: [Int] -> Bool
skontrRiadok riadok = (length riadok) == (length $ nub riadok)

skontrSudoku :: [[Int]] -> Bool
skontrSudoku sud = null $ filter (\x -> not x) [skontrRiadok r | r <- sud]

testSudokuStvorce :: [[Int]] -> Bool
testSudokuStvorce sud = (skontrSudoku sud) && (skontrSudoku (transpose sud)) &&  (skontrSudoku (sudokuStvorce sud))
