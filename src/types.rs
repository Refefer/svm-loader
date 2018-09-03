use std::fmt::Debug;
/// Defines datastypes

/// Sparse datatype
#[derive(Debug,Clone)]
pub struct Sparse(pub usize, pub Vec<usize>, pub Vec<f32>);

impl Sparse {
    pub fn to_dense(&self) -> Vec<f32> {
        let mut v = vec![0f32; self.0];
        for idx in 0..self.1.len() {
            v[self.1[idx]] = self.2[idx];
        }
        v
    }
}

pub trait DataParse {
    type Out: Debug;

    fn parse<'a, I: Iterator<Item=&'a str>>(&self, xs: I) -> Option<Self::Out>;
}

#[derive(Debug)]
pub struct DenseData;

impl DataParse for DenseData {
    type Out = Vec<f32>;

    fn parse<'a, I: Iterator<Item=&'a str>>(&self, xs: I) -> Option<Self::Out> {
        xs.map(|x| {
            x.split(':').last().and_then(|x| x.parse().ok())
        }).collect()
    }
}

#[derive(Debug)]
pub struct SparseData(pub usize);

impl DataParse for SparseData {
    type Out = Sparse;

    fn parse<'a, I: Iterator<Item=&'a str>>(&self, xs: I) -> Option<Self::Out> {
        let ivs: Option<Vec<(usize,f32)>> = xs.map(|x| {
            let mut p = x.split(':');
            let idx: Option<usize> = p.next()
                .and_then(|idx| idx.parse().ok());
            let v: Option<f32> = p.next()
                .and_then(|val| val.parse().ok());

            idx.and_then(|i| v.map(|vi| (i, vi)))
        }).collect();

        ivs.map(|mut iv| {
            // Sort then dedup by key
            iv.sort_by_key(|x| x.0);
            iv.dedup_by_key(|x| x.0);
            let (is, vs) = iv.into_iter()
                .filter(|x| x.0 < self.0 && x.1 != 0.0).unzip();

            Sparse(self.0, is, vs)
        })
    }
}

pub trait Dimension {
    type Out;
    fn dims(&self) -> Self::Out;
}

impl Dimension for Vec<f32> {
    type Out = usize;
    fn dims(&self) -> Self::Out { self.len() }
}

impl Dimension for Sparse {
    type Out = usize;
    fn dims(&self) -> Self::Out { self.0 }
}
