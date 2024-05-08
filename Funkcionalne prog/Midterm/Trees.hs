module Trees where

-- Marian Kravec

data BTree t = BNode (BTree t) t (BTree t) | Nil deriving (Show, Eq)

bserialize :: BTree t->[Maybe t]
bserialize (BNode l v r) = [Just v]++(bserialize l)++(bserialize r)
bserialize Nil = [Nothing]

bdeserialize :: [Maybe t]->BTree t
bdeserialize xs = fst (deset xs)
                  where
                    deser :: [Maybe t] -> (BTree t, [Maybe t])
                    deser [] = (Nil, [])
                    deset ((Nothing):xs) = (Nil, xs)
                    deset ((Just v):xs) = let
                                             (l, rest1) = deset xs
                                             (r, rest2) = deset rest1
                                          in
                                             ((BNode l v r), rest2)


t = BNode (BNode Nil 3 Nil) 5 (BNode Nil 6 Nil) :: BTree Int
test_t = t == (bdeserialize (bserialize t))


data Tree t = Node t [Tree t] deriving (Show)

serialize::Tree t->[Maybe t]
serialize (Node v xs) = [Just v]++(concatMap (serialize) xs)++[Nothing]

deserialize::[Maybe t]->Tree t
deserialize ((Just v):xs) = let
                           temp_tree = (Node v [])
                         in
                           fst (deset (temp_tree, xs))
                         where
                           deser :: (Tree t, [Maybe t]) -> (Tree t, [Maybe t])
                           deser (t, []) = (t, [])
                           deset (t, ((Nothing):xs)) = (t, xs)
                           deset ((Node v ts), ((Just newV):xs)) = let
                                                        newT = (Node newV [])
                                                        (t, rest) = deset (newT, xs)
                                                      in
                                                        deset ((Node v (ts++[t])), rest)

s = Node 3 [Node 2 [], Node 5 [], Node 7 []] :: Tree Int