module Rozcv_11 where

import Control.Monad.State

-- Marian Kravec

data Tree a = Node a [Tree a]  deriving (Show, Eq)

e1 :: Tree String
e1 = Node "Jano" [    (Node "Fero"  [(Node "a" []),(Node "b" []),(Node "a" []),(Node "b" [])]),
            (Node "Jano"  [(Node "a" [])]),
            (Node "Karel"  [(Node "c" []),(Node "a" []),(Node "c" [])]),
            (Node "Fero"  [(Node "d" []),(Node "b" []),(Node "a" []),(Node "c" [])]),
            (Node "Karel"  [(Node "d" []),(Node "a" []),(Node "d" [])])
        ]

size :: Tree a -> Int
size (Node a ta) = 1 + (foldl (\acc x -> acc + (size x)) 0 ta)

reindex :: Tree a -> Tree Int
reindex t = evalState (reindexState t) 0

reindexState :: Tree a -> State Int (Tree Int)
reindexState (Node a ta) = do
                             s <- get
                             put (s+1)
                             newTa <- mapM reindexState ta
                             return (Node s newTa)

rename  :: (Eq a) => Tree a -> Tree Int
rename t = evalState (renameState t) []

inMap :: (Eq a) => [a] -> a -> Int -> Int
inMap [] _ _ = -1
inMap (x:xs) a i | x == a = i
                 | otherwise = inMap xs a (i+1)


renameState :: (Eq a) => Tree a -> State [a] (Tree Int)
renameState (Node a ta) = do
                             s <- get
                             poz <- return (inMap s a 0)
                             l <- return (length s)
                             put (if poz == -1 then (s++[a]) else s)
                             i <- return (if poz == -1 then l else poz)
                             newTa <- mapM renameState ta
                             return (Node i newTa)