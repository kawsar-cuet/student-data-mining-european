"""
LLM-based Recommendation Engine
Generates personalized recommendations for students using Large Language Models
"""

import os
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class RecommendationEngine:
    """
    Generate personalized student recommendations using LLM
    Supports both OpenAI API and local models
    """
    
    def __init__(self, api_key=None, model="gpt-4", temperature=0.7):
        """
        Initialize the recommendation engine
        
        Args:
            api_key: OpenAI API key (optional, can use environment variable)
            model: LLM model to use
            temperature: Sampling temperature for generation
        """
        self.model = model
        self.temperature = temperature
        
        # Try to get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        
        self.api_key = api_key
        self.client = None
        
        # Initialize OpenAI client if API key is available
        if self.api_key and self.api_key != 'your-openai-api-key-here':
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                print(f"✓ Recommendation Engine initialized with {model}")
            except ImportError:
                print("⚠ OpenAI library not installed. Install with: pip install openai")
                self.client = None
            except Exception as e:
                print(f"⚠ Could not initialize OpenAI client: {str(e)}")
                self.client = None
        else:
            print("⚠ No OpenAI API key provided. Using rule-based recommendations.")
    
    def create_student_profile(self, student_data: Dict, 
                              predicted_grade: str, 
                              dropout_prob: float) -> str:
        """
        Create a comprehensive student profile for LLM input
        
        Args:
            student_data: Dictionary containing student information
            predicted_grade: Predicted final grade
            dropout_prob: Dropout probability (0-1)
            
        Returns:
            Formatted student profile string
        """
        risk_level = self._get_risk_level(dropout_prob)
        
        profile = f"""
Student Profile:
- Name: {student_data.get('name', 'N/A')}
- Department: {student_data.get('department', 'N/A')}
- Semester: {student_data.get('semester', 'N/A')}
- Age: {student_data.get('age', 'N/A')}

Academic Performance:
- Current CGPA: {student_data.get('cgpa', 'N/A')}
- Previous Semester CGPA: {student_data.get('previous_semester_cgpa', 'N/A')}
- Predicted Final Grade: {predicted_grade}
- Midterm Score: {student_data.get('midterm_score', 'N/A')}/100
- Quiz Average: {student_data.get('quiz_average', 'N/A')}/100
- Assignment Submission Rate: {student_data.get('assignment_submission_rate', 'N/A')}%

Dropout Risk Assessment:
- Dropout Probability: {dropout_prob:.2%}
- Risk Level: {risk_level}

Engagement Metrics:
- Attendance Rate: {student_data.get('attendance_rate', 'N/A')}%
- Study Hours per Week: {student_data.get('study_hours_per_week', 'N/A')}
- Library Visits per Month: {student_data.get('library_visits_per_month', 'N/A')}
- Participation Score: {student_data.get('participation_score', 'N/A')}/100

Behavioral Factors:
- Sleep Hours: {student_data.get('sleep_hours', 'N/A')}
- Social Media Hours: {student_data.get('social_media_hours', 'N/A')}
- Stress Level: {student_data.get('stress_level', 'N/A')}
- Motivation Level: {student_data.get('motivation_level', 'N/A')}

Support & Resources:
- Scholarship: {student_data.get('scholarship', 'N/A')}
- Mentor Meetings: {student_data.get('mentor_meetings', 'N/A')} per month
- Part-time Job: {student_data.get('part_time_job', 'N/A')}
- Internet Access: {student_data.get('internet_access', 'N/A')}
- Health Issues: {student_data.get('health_issues', 'N/A')}

Key Challenges Identified:
{self._identify_challenges(student_data, dropout_prob)}
"""
        return profile
    
    def _get_risk_level(self, dropout_prob: float) -> str:
        """Categorize dropout risk"""
        if dropout_prob < 0.3:
            return "Low Risk"
        elif dropout_prob < 0.7:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def _identify_challenges(self, student_data: Dict, dropout_prob: float) -> str:
        """Identify key challenges from student data"""
        challenges = []
        
        # Academic challenges
        if student_data.get('cgpa', 4.0) < 2.5:
            challenges.append("- Low CGPA (below 2.5)")
        
        if student_data.get('attendance_rate', 100) < 70:
            challenges.append("- Poor attendance (below 70%)")
        
        if student_data.get('assignment_submission_rate', 100) < 75:
            challenges.append("- Low assignment submission rate")
        
        # Behavioral challenges
        if student_data.get('study_hours_per_week', 30) < 15:
            challenges.append("- Insufficient study time")
        
        if student_data.get('sleep_hours', 8) < 6:
            challenges.append("- Sleep deprivation (less than 6 hours)")
        
        if student_data.get('stress_level') == 'High':
            challenges.append("- High stress levels")
        
        if student_data.get('motivation_level') == 'Low':
            challenges.append("- Low motivation")
        
        # Social challenges
        if student_data.get('social_media_hours', 0) > 5:
            challenges.append("- Excessive social media usage")
        
        # Resource challenges
        if student_data.get('internet_access') == 'No':
            challenges.append("- Limited internet access")
        
        if student_data.get('part_time_job') == 'Yes':
            challenges.append("- Balancing work and studies")
        
        # Dropout risk
        if dropout_prob > 0.7:
            challenges.append("- HIGH DROPOUT RISK - Immediate intervention needed")
        elif dropout_prob > 0.3:
            challenges.append("- Moderate dropout risk - Early support recommended")
        
        return "\n".join(challenges) if challenges else "- No major challenges identified"
    
    def generate_recommendations(self, student_data: Dict, 
                                predicted_grade: str,
                                dropout_prob: float,
                                max_tokens: int = 500) -> str:
        """
        Generate personalized recommendations using LLM
        
        Args:
            student_data: Student information dictionary
            predicted_grade: Predicted final grade
            dropout_prob: Dropout probability
            max_tokens: Maximum tokens for generation
            
        Returns:
            Personalized recommendations string
        """
        profile = self.create_student_profile(student_data, predicted_grade, dropout_prob)
        
        # If OpenAI client is available, use it
        if self.client:
            return self._generate_with_openai(profile, max_tokens)
        else:
            # Fallback to rule-based recommendations
            return self._generate_rule_based(student_data, dropout_prob)
    
    def _generate_with_openai(self, profile: str, max_tokens: int) -> str:
        """Generate recommendations using OpenAI API"""
        
        system_prompt = """You are an experienced academic advisor at the University of Liberal Arts Bangladesh (ULAB). 
Your role is to analyze student profiles and provide specific, actionable, and empathetic recommendations 
to help students improve their academic performance and reduce dropout risk.

Focus on:
1. Academic strategies (study techniques, time management, course selection)
2. Behavioral improvements (sleep, stress management, focus)
3. Resource utilization (library, mentors, scholarships, counseling)
4. Social support (peer groups, family communication)
5. Health and wellness (physical and mental health)

Provide 3-5 concrete, personalized recommendations prioritized by impact."""

        user_prompt = f"""{profile}

Based on this student profile, provide personalized recommendations to help them improve their 
academic performance and reduce dropout risk. Be specific and actionable."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            
            recommendations = response.choices[0].message.content
            return recommendations
            
        except Exception as e:
            print(f"⚠ Error calling OpenAI API: {str(e)}")
            return self._generate_rule_based(student_data={}, dropout_prob=0.5)
    
    def _generate_rule_based(self, student_data: Dict, dropout_prob: float) -> str:
        """Generate rule-based recommendations as fallback"""
        
        recommendations = ["PERSONALIZED RECOMMENDATIONS:\n"]
        
        # High dropout risk
        if dropout_prob > 0.7:
            recommendations.append("""
1. URGENT: Schedule an immediate meeting with your academic advisor
   - Discuss your current challenges and create an intervention plan
   - Explore options for academic support and counseling services
   - Consider course load adjustment if needed
""")
        
        # Academic recommendations
        if student_data.get('cgpa', 4.0) < 2.5:
            recommendations.append("""
2. Academic Recovery Plan:
   - Attend all remaining classes and actively participate
   - Form study groups with high-performing peers
   - Visit professors during office hours for clarification
   - Consider tutoring services for challenging subjects
""")
        
        # Attendance issues
        if student_data.get('attendance_rate', 100) < 70:
            recommendations.append("""
3. Improve Attendance:
   - Set multiple alarms and prepare the night before
   - Sit in front rows to stay engaged
   - Track attendance weekly to monitor progress
   - Address any health or transportation issues affecting attendance
""")
        
        # Time management
        if student_data.get('study_hours_per_week', 30) < 15:
            recommendations.append("""
4. Time Management:
   - Create a weekly study schedule with dedicated time blocks
   - Use Pomodoro technique (25 min study + 5 min break)
   - Reduce social media usage during study hours
   - Prioritize assignments by deadline and importance
""")
        
        # Stress and wellness
        if student_data.get('stress_level') == 'High' or student_data.get('sleep_hours', 8) < 6:
            recommendations.append("""
5. Health and Wellness:
   - Aim for 7-8 hours of sleep nightly
   - Practice stress management (meditation, exercise, hobbies)
   - Utilize university counseling services if available
   - Maintain a balanced diet and regular exercise routine
""")
        
        # Resource utilization
        recommendations.append("""
6. Leverage Available Resources:
   - Increase library visits for focused study environment
   - Attend mentor meetings regularly (aim for 2-3 per month)
   - Join relevant academic clubs and study groups
   - Explore scholarship opportunities to reduce financial stress
""")
        
        # Positive reinforcement
        if dropout_prob < 0.3:
            recommendations.append("""
7. Maintain Your Success:
   - Continue your current positive habits
   - Consider peer tutoring to reinforce your knowledge
   - Explore leadership opportunities in student organizations
   - Set new academic goals to stay challenged and motivated
""")
        
        return "\n".join(recommendations[:6])  # Limit to top 6 recommendations
    
    def generate_batch_recommendations(self, students_data: List[Dict],
                                      predicted_grades: List[str],
                                      dropout_probs: List[float]) -> List[str]:
        """
        Generate recommendations for multiple students
        
        Args:
            students_data: List of student data dictionaries
            predicted_grades: List of predicted grades
            dropout_probs: List of dropout probabilities
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        print(f"\nGenerating recommendations for {len(students_data)} students...")
        
        for i, (student, grade, prob) in enumerate(zip(students_data, predicted_grades, dropout_probs)):
            print(f"Processing student {i+1}/{len(students_data)}...", end='\r')
            rec = self.generate_recommendations(student, grade, prob)
            recommendations.append(rec)
        
        print(f"\n✓ Generated {len(recommendations)} recommendations")
        
        return recommendations


if __name__ == "__main__":
    # Test the recommendation engine
    engine = RecommendationEngine()
    
    # Sample student data
    sample_student = {
        'name': 'Ahmed Rahman',
        'department': 'CSE',
        'semester': 3,
        'age': 20,
        'cgpa': 3.45,
        'previous_semester_cgpa': 3.38,
        'midterm_score': 78,
        'quiz_average': 82,
        'assignment_submission_rate': 92,
        'attendance_rate': 85,
        'study_hours_per_week': 25,
        'library_visits_per_month': 8,
        'participation_score': 75,
        'sleep_hours': 7,
        'social_media_hours': 4,
        'stress_level': 'Medium',
        'motivation_level': 'High',
        'scholarship': 'Yes',
        'mentor_meetings': 3,
        'part_time_job': 'No',
        'internet_access': 'Yes',
        'health_issues': 'No'
    }
    
    recommendations = engine.generate_recommendations(
        sample_student,
        predicted_grade='A-',
        dropout_prob=0.15
    )
    
    print(recommendations)
