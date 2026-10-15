//! Sample mapping passed across the crate boundary.
//!
//! Lisca builds this from `assay.json` and clones the fields. The two
//! `SampleMapping` types stay distinct so this crate does not depend on lisca.

use std::collections::{BTreeMap, HashMap, HashSet};

/// Analysis settings for one Sample: its Positions and channel roles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleAnalysis {
    pub name: String,
    pub positions: Vec<u32>,
    pub signal: Vec<u32>,
    pub segmentation: u32,
}

/// Samples in assay order. Stages address a Sample by its index here.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SampleMapping(pub Vec<SampleAnalysis>);

impl SampleMapping {
    pub fn iter(&self) -> std::slice::Iter<'_, SampleAnalysis> {
        self.0.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Distinct Positions across all Samples, in assay order.
    pub fn positions(&self) -> Vec<u32> {
        let mut seen = HashSet::new();
        self.0
            .iter()
            .flat_map(|sample| sample.positions.iter().copied())
            .filter(|position| seen.insert(*position))
            .collect()
    }

    /// The Samples that include `position`, each narrowed to that Position.
    pub fn for_position(&self, position: u32) -> SampleMapping {
        SampleMapping(
            self.0
                .iter()
                .filter(|sample| sample.positions.contains(&position))
                .map(|sample| SampleAnalysis {
                    positions: vec![position],
                    ..sample.clone()
                })
                .collect(),
        )
    }
}

impl FromIterator<SampleAnalysis> for SampleMapping {
    fn from_iter<T: IntoIterator<Item = SampleAnalysis>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<'a> IntoIterator for &'a SampleMapping {
    type Item = &'a SampleAnalysis;
    type IntoIter = std::slice::Iter<'a, SampleAnalysis>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// `results/<dirname>/` per Sample index. Names that sanitize to the same
/// dirname are prefixed with their 0-based assay index.
pub fn sample_pack_dirnames(mapping: &SampleMapping) -> BTreeMap<usize, String> {
    let sanitized: BTreeMap<usize, String> = mapping
        .iter()
        .enumerate()
        .map(|(index, sample)| (index, filesystem_safe_sample_name(&sample.name)))
        .collect();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for name in sanitized.values() {
        *counts.entry(name.clone()).or_default() += 1;
    }
    sanitized
        .into_iter()
        .map(|(index, name)| {
            let dirname = if counts.get(&name).copied().unwrap_or(0) > 1 {
                format!("{index}_{name}")
            } else {
                name
            };
            (index, dirname)
        })
        .collect()
}

fn filesystem_safe_sample_name(name: &str) -> String {
    let mut text = String::new();
    let mut last_underscore = false;
    for ch in name.trim().chars() {
        let replacement = match ch {
            '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*' => Some('_'),
            c if c.is_control() => Some('_'),
            c if c.is_whitespace() => Some('_'),
            _ => None,
        };
        if let Some(repl) = replacement {
            if !last_underscore && !text.is_empty() {
                text.push(repl);
                last_underscore = true;
            }
        } else {
            text.push(ch);
            last_underscore = false;
        }
    }
    let trimmed = text.trim_matches(|c: char| c == '_' || c == '.' || c == ' ');
    let trimmed = trimmed.trim_start_matches('.');
    if trimmed.is_empty() {
        "sample".to_string()
    } else {
        trimmed.to_string()
    }
}
