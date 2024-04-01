module MinMax where
data BTree t = Node (BTree t) t (BTree t) | Nil deriving(Show, Eq)

-- Marian Kravec

minmax :: (Ord t) => (BTree t) -> (t, t)
minmax tree = ((minBTree tree), (maxBTree tree))
              where
                maxBTree :: (Ord t) => (BTree t) -> t

                maxBTree (Node l v r) | (l == Nil) && (r == Nil) = v
                                      | l == Nil = max v (maxBTree r)
                                      | r == Nil = max (maxBTree l) v
                                      | otherwise = max (max (maxBTree l) v) (maxBTree r)

                minBTree :: (Ord t) => (BTree t) -> t
                minBTree (Node l v r) | (l == Nil) && (r == Nil) = v
                                      | l == Nil = min v (minBTree r)
                                      | r == Nil = min (minBTree l) v
                                      | otherwise = min (min (minBTree l) v) (minBTree r)