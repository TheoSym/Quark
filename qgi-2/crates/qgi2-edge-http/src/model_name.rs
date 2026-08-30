//! Persona encoding in the model name.
//!
//! An OpenAI-compatible client has exactly one field it can use to pick a
//! behaviour: `model`. QGI-2 encodes the persona there as
//! `qgi2/<mood>-<profile>`, so a caller that knows nothing about moods still
//! gets a valid one, and a caller that does can switch persona with its
//! existing model switcher.

use qgi2_spec_types::{Mood, Persona, Profile};

/// A parsed model name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelName {
    pub persona: Persona,
}

impl ModelName {
    pub fn render(&self) -> String {
        format!(
            "qgi2/{}-{}",
            self.persona.mood.as_str(),
            self.persona.profile.as_str()
        )
    }
}

/// Every persona, for the `/v1/models` listing.
pub fn all_model_names() -> Vec<String> {
    let mut out = Vec::new();
    for mood in Mood::ALL {
        for profile in Profile::ALL {
            out.push(
                ModelName {
                    persona: Persona::new(mood, profile),
                }
                .render(),
            );
        }
    }
    out
}

/// Parse `qgi2/<mood>-<profile>`.
///
/// Unknown or partial names fall back to the default persona rather than
/// failing the request: a client that has not been told about the naming
/// convention should still get a working agent, and the persona it got is
/// echoed back in the response's `model` field so the fallback is visible
/// rather than silent.
pub fn parse_model_name(name: &str) -> ModelName {
    let stripped = name.strip_prefix("qgi2/").unwrap_or(name);

    // `split_once` on the *first* hyphen: moods and profiles contain none, so
    // anything after the second is junk to ignore rather than a parse failure.
    let (mood_part, profile_part) = match stripped.split_once('-') {
        Some((m, p)) => (m, p),
        None => (stripped, ""),
    };

    let mood = mood_part.parse::<Mood>().unwrap_or(Mood::Builder);
    let profile = profile_part.parse::<Profile>().unwrap_or(Profile::Traceable);

    ModelName {
        persona: Persona::new(mood, profile),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_name_parses() {
        let m = parse_model_name("qgi2/researcher-deterministic");
        assert_eq!(m.persona.mood, Mood::Researcher);
        assert_eq!(m.persona.profile, Profile::Deterministic);
    }

    #[test]
    fn the_prefix_is_optional() {
        assert_eq!(
            parse_model_name("companion-quick").persona,
            Persona::new(Mood::Companion, Profile::Quick)
        );
    }

    #[test]
    fn an_unknown_name_falls_back_to_the_default_persona() {
        // A client that has not been told the convention still gets a working
        // agent; the response echoes which persona it actually got.
        assert_eq!(parse_model_name("gpt-4o").persona.mood, Mood::Builder);
        assert_eq!(parse_model_name("").persona, Persona::default());
    }

    #[test]
    fn a_mood_without_a_profile_gets_the_default_profile() {
        let m = parse_model_name("qgi2/companion");
        assert_eq!(m.persona.mood, Mood::Companion);
        assert_eq!(m.persona.profile, Profile::Traceable);
    }

    #[test]
    fn every_persona_round_trips() {
        for name in all_model_names() {
            let parsed = parse_model_name(&name);
            assert_eq!(parsed.render(), name);
        }
    }

    #[test]
    fn the_listing_covers_every_combination() {
        assert_eq!(all_model_names().len(), Mood::ALL.len() * Profile::ALL.len());
    }
}
