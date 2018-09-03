pub mod types;

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader,BufRead,Error};

use types::DataParse;

pub trait TargetReader {
    type Out;

    fn process(&self, data: &str) -> Option<Self::Out>;
}

pub struct Regression;

impl TargetReader for Regression {
    type Out = f32;

    fn process(&self, data: &str) -> Option<Self::Out> {
        data.parse().ok()
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
    pub x: F,
    pub y: T,
    pub qid: Option<usize>,
    pub comment: Option<String>,
}

impl <T,F> Row<T,F> {
    pub fn new(x: F, y: T, qid: Option<usize>, comment: Option<String>) -> Self {
        Row {
            x: x,
            y: y,
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
                let res = parse_line(self.tr, self.p, &self.tl)
                    .map(|x| Row::new(x.1, x.0, None, None));

                if res.is_some() { return res }

            } else { 
                return None 
            }
        }
    }
}

fn parse_line<TR: TargetReader, DP: DataParse>(tr: &TR, dp: &DP, line: &str) -> Option<(TR::Out, DP::Out)> {
    let has_target = !line.starts_with(' ');
    // Remove comments
    let line = line.split('#').next().unwrap();
    let mut pieces = line.trim().split_whitespace();
    let target = if has_target {
        pieces.next().and_then(|x| tr.process(x))
    } else {
        tr.process("")
    };

    let vec = dp.parse(pieces);

    match (target, vec) {
        (Some(s), Some(v)) => Some((s,  v)),
        _ => None
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}