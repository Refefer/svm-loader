pub mod types;

use std::fmt::Debug;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader,BufRead,Error};

use types::DataParse;

pub trait TargetReader {
    type Out: Debug;

    fn process(&self, data: &str) -> Option<Self::Out>;
}

pub struct Regression;

impl TargetReader for Regression {
    type Out = f32;

    fn process(&self, data: &str) -> Option<Self::Out> {
        data.parse().ok()
    }
}

pub struct BinaryClassification;

impl TargetReader for BinaryClassification {
    type Out = bool;

    fn process(&self, data: &str) -> Option<Self::Out> {
        match data {
            "-1" => Some(false),
            "1"  => Some(true),
            _    => None
        }
    }
}

pub struct DisjointClassification;

impl TargetReader for DisjointClassification {
    type Out = usize;

    fn process(&self, data: &str) -> Option<Self::Out> {
        data.parse().ok()
    }
}

pub struct MultiLabelClassification;

impl TargetReader for MultiLabelClassification {
    type Out = HashSet<usize>;

    fn process(&self, data: &str) -> Option<Self::Out> {
        let mut classes = HashSet::new();
        for piece in data.split(',') {
            if let Ok(cid) = piece.parse() {
                classes.insert(cid);
            }
        }
        Some(classes)
    }
}

pub struct Tags;

impl TargetReader for Tags {
    type Out = HashSet<String>;

    fn process(&self, data: &str) -> Option<Self::Out> {
        let mut classes = HashSet::new();
        for piece in data.split(',') {
            if let Ok(tag) = piece.parse() {
                classes.insert(tag);
            }
        }
        classes.remove("");
        Some(classes)
    }
}


pub struct Row<T,F> {
    pub y: T,
    pub x: F,
    pub qid: Option<usize>,
    pub comment: Option<String>,
}

impl <T,F> Row<T,F> {
    pub fn new(y: T, x: F, qid: Option<usize>, comment: Option<String>) -> Self {
        Row {
            y: y,
            x: x,
            qid: qid,
            comment: comment
        }
    }
}

pub fn load<'a, TR: TargetReader, P: DataParse>(fname: &str, tr: &'a TR, p: &'a P) -> Result<Reader<'a, TR,P>,Error> {
    let f = File::open(fname)?;
    let br = BufReader::new(f);
    Ok(Reader {br: br, p: p, tr: tr, tl: String::new()})
}

pub struct Reader<'a, TR: 'a + TargetReader,P: 'a + DataParse> {
    br: BufReader<File>,
    p: &'a P,
    tr: &'a TR,
    tl: String
}

impl <'a, TR: 'a + TargetReader, P: 'a + DataParse> Iterator for Reader<'a, TR, P> {
    type Item = Row<TR::Out, P::Out>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.tl.clear();
            if let Ok(size) = self.br.read_line(&mut self.tl) {
                if size == 0 { return None }
                let res = parse_line(self.tr, self.p, &self.tl);

                if res.is_some() { return res }

            } else { 
                return None 
            }
        }
    }
}

struct IterCons<X,I>(Option<X>, I);

impl <X, I: Iterator<Item=X>> Iterator for IterCons<X, I> {
    type Item = X;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0.is_some() {
            self.0.take()
        } else {
            self.1.next()
        }
    }
}

pub fn parse_line<TR: TargetReader, DP: DataParse>(tr: &TR, dp: &DP, line: &str) -> Option<Row<TR::Out,DP::Out>> {
    let has_target = !line.starts_with(' ');
    // Remove comments
    let mut data = line.split('#');
    let line = data.next().unwrap();
    let comment = data.next().map(|x| x.to_owned());
    let mut pieces = line.trim().split_whitespace();
    let target = if has_target {
        pieces.next().and_then(|x| tr.process(x))
    } else {
        tr.process("")
    };

    // Check for qid
    let maybe_qid = pieces.next();
    let qid: Option<usize> = maybe_qid.and_then(|qid| {
        if qid.starts_with("qid:") {
            let mut p = qid.split(':').skip(1);
            p.next().unwrap().parse().ok()
        } else {
            None
        }
    });
    let peeked = if qid.is_some() {
        IterCons(None, pieces)
    } else {
        IterCons(maybe_qid, pieces)
    };

    let vec = dp.parse(peeked);

    match (target, vec) {
        (Some(y), Some(x)) => Some(Row::new(y, x, qid, comment)),
        _ => None
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use types::*;
    #[test]
    fn parse_line_1() {
        let sd = SparseData(12);
        let td = DisjointClassification;

        let s = "1 qid:1234 0:-13 11:10 # hello";
        let srow = parse_line(&td, &sd, s);
        assert!(srow.is_some());
        let row = srow.unwrap();

        assert_eq!(row.y, 1usize);
        assert_eq!(row.qid, Some(1234));
        assert_eq!(row.comment, Some(" hello".into()));
    }

    fn parse_bool_1() {
        let sd = SparseData(12);
        let td = BinaryClassification;

        let s2 = "-1 qid:1234 0:-13 11:10 # hello";
        let srow = parse_line(&td, &sd, s2);
        assert!(srow.is_some());
        let row = srow.unwrap();

        assert_eq!(row.y, false);
        assert_eq!(row.qid, Some(1234));
        assert_eq!(row.comment, Some(" hello".into()));

    }
}
