import unittest

from qald.grounding import Linker, GetURIType

class GetURITypeTestCase(unittest.TestCase):
  def setUp(self):
    self.linker = Linker('.uri_cache_test')

  @unittest.expectedFailure
  def test_EntityPersonProfessor(self):
    uri = 'fb:en.professor'
    self.assertEqual("person", GetURIType(uri))

  def test_PredicatePersonAcademicPerson(self):
    uri = 'fb:education.academic_post.person'
    self.assertEqual("person", GetURIType(uri))

  @unittest.expectedFailure
  def test_PredicateLocationInstitution(self):
    uri = 'fb:education.academic_post.institution'
    self.assertEqual("location", GetURIType(uri))

  def test_PredicateLocationInstitution(self):
    uri = 'fb:location.mailing_address.citytown'
    self.assertEqual("location", GetURIType(uri))

  def test_PredicateLocationRegions(self):
    uri = 'fb:finance.stock_exchange.primary_regions'
    self.assertEqual("location", GetURIType(uri))

  def test_PredicateNumberIssueNumber(self):
    uri = 'fb:comic_books.comic_book_issue.issue_number'
    self.assertEqual("number", GetURIType(uri))

  def test_PredicatePersonSurvivors(self):
    uri = 'fb:event.disaster.survivors'
    # from pudb import set_trace; set_trace()
    self.assertEqual("person", GetURIType(uri))

  @unittest.expectedFailure
  def test_PredicatePersonCelebrity(self):
    uri = 'fb:celebrities.rehab.celebrity'
    self.assertEqual("person", GetURIType(uri))

  def test_PredicatePersonPlayer(self):
    uri = 'fb:american_football.football_roster_position.player'
    self.assertEqual("person", GetURIType(uri))

  def test_PredicatePersonEmploymentPerson(self):
    uri = 'fb:business.employment_tenure.person'
    self.assertEqual("person", GetURIType(uri))

  def test_PredicatePersonCrewmember(self):
    uri = 'fb:film.film_crew_gig.crewmember'
    self.assertEqual("person", GetURIType(uri))

  def test_PredicatePersonManager(self):
    uri = 'fb:soccer.football_team_management_tenure.manager'
    self.assertEqual("person", GetURIType(uri))

  def test_PredicatePersonPractitioner(self):
    uri = 'fb:martial_arts.martial_art.well_known_practitioner'
    # from pudb import set_trace; set_trace()
    self.assertEqual("person", GetURIType(uri))

  def test_PredicatePersonNoFuel(self):
    uri = 'fb:engineering.engine.energy_source'
    # from pudb import set_trace; set_trace()
    self.assertNotEqual("person", GetURIType(uri))

  def test_PredicateNumberYears(self):
    uri = 'fb:games.game.minimum_age_years'
    self.assertEqual("number", GetURIType(uri))

  def test_PredicateNumberDuration(self):
    uri = 'fb:amusement_parks.ride.duration'
    self.assertEqual("number", GetURIType(uri))

  def test_PredicateNumberAmount(self):
    uri = 'fb:measurement_unit.dated_money_value.amount'
    self.assertEqual("number", GetURIType(uri))

  def test_PredicateNumberTemperature(self):
    uri = 'fb:travel.travel_destination_monthly_climate.average_max_temp_c'
    self.assertEqual("number", GetURIType(uri))

  def test_PredicateNumberHeight(self):
    uri = 'fb:architecture.structure.height_meters'
    self.assertEqual("number", GetURIType(uri))

  def test_PredicateDateCompletedOn(self):
    uri = 'fb:law.constitutional_amendment.ratification_completed_on'
    self.assertEqual("date", GetURIType(uri))

  def test_PredicateDatePublicationDate(self):
    uri = 'fb:book.written_work.date_of_first_publication'
    self.assertEqual("date", GetURIType(uri))

  def test_PredicateDateDuringPeriod(self):
    uri = 'fb:geology.geological_formation.formed_during_period'
    self.assertEqual("date", GetURIType(uri))

  def test_PredicateDateLastEruption(self):
    uri = 'fb:geography.mountain.last_eruption'
    self.assertEqual("date", GetURIType(uri))

  def test_PredicateDateFoundedDate(self):
    uri = 'fb:organization.organization.date_founded'
    self.assertEqual("date", GetURIType(uri))

class IsDisambiguatorTestCase(unittest.TestCase):
  def setUp(self):
    self.linker = Linker('.uri_cache_test')

  def test_RelPositive(self):
    rel = 'fb:education.academic_post.person'
    self.assertTrue(self.linker.IsURIDisambiguator(rel))

  def test_RelNegative(self):
    rel = 'fb:automotive.trim_level.max_passengers'
    self.assertFalse(self.linker.IsURIDisambiguator(rel))

  def test_RelNegative2(self):
    rel = 'fb:book.technical_report.institution'
    self.assertFalse(self.linker.IsURIDisambiguator(rel))

  def test_EntNegative(self):
    rel = 'fb:en.marshall_hall'
    self.assertFalse(self.linker.IsURIDisambiguator(rel))

if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(IsDisambiguatorTestCase)
  suite2  = unittest.TestLoader().loadTestsFromTestCase(GetURITypeTestCase)
  suites  = unittest.TestSuite([suite1, suite2])
  unittest.TextTestRunner(verbosity=2).run(suites)

