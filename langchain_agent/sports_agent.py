import os
import requests
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_utils import init_gemini_model

llm = init_gemini_model()

API_KEY  = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS  = {
    "x-apisports-key": API_KEY
}

@tool
def get_live_scores(league: str = "all") -> str:
    """Get live football scores. Input: league ID or 'all'"""
    params = {"live": "all"}
    if league.lower() != "all":
        params["league"] = league
    data = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params=params).json()
    fixtures = data.get("response", [])
    if not fixtures:
        return "No live matches right now."
    result = ""
    for f in fixtures[:10]:
        result += f"{f['teams']['home']['name']} {f['goals']['home']} - {f['goals']['away']} {f['teams']['away']['name']} | {f['league']['name']}\n"
    return result

@tool
def get_recent_matches(team_name: str) -> str:
    """Get last 5 match results for a team"""
    response = requests.get(f"{BASE_URL}/teams", headers=HEADERS, params={"search": team_name}).json()
    if not response.get("response"):
        return f"Team '{team_name}' not found."
    team_id  = response["response"][0]["team"]["id"]
    fixtures = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params={"team": team_id, "last": 5}).json().get("response", [])
    if not fixtures:
        return f"No recent matches found for {team_name}."
    result = ""
    for f in fixtures:
        result += f"{f['fixture']['date'][:10]} | {f['teams']['home']['name']} {f['goals']['home']} - {f['goals']['away']} {f['teams']['away']['name']}\n"
    return result

@tool
def get_standings(league_id: str) -> str:
    """Get league standings. Input: league_id like '39' for Premier League"""
    data = requests.get(f"{BASE_URL}/standings", headers=HEADERS, params={"league": league_id, "season": "2024"}).json()
    if not data.get("response"):
        return "No standings found."
    standings = data["response"][0]["league"]["standings"][0]
    result    = ""
    for t in standings[:5]:
        result += f"{t['rank']}. {t['team']['name']} - {t['points']} pts\n"
    return result

@tool
def get_league_teams(league_id: str) -> str:
    """Get all teams in a league (e.g., '39' for Premier League)."""
    data = requests.get(f"{BASE_URL}/teams", headers=HEADERS, params={"league": league_id, "season": "2024"}).json()
    teams = data.get("response", [])
    if not teams:
        return f"No teams found for league {league_id}."
    names = [t['team']['name'] for t in teams]
    return "League teams:\n" + "\n".join(sorted(names)) + f"\nTotal teams: {len(names)}"

@tool
def get_upcoming_matches(league_id: str = "all") -> str:
    """Get upcoming football matches (next 10) and list teams."""
    params = {"status": "NS", "next": 10}
    if league_id.lower() != "all":
        params["league"] = league_id
    data = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params=params).json()
    fixtures = data.get("response", [])
    if not fixtures:
        return "No upcoming matches found."

    team_names = set()
    result = "Upcoming matches:\n"
    for f in fixtures:
        home = f['teams']['home']['name']
        away = f['teams']['away']['name']
        date = f['fixture']['date'][:16].replace('T', ' ')
        league = f['league']['name']
        team_names.add(home)
        team_names.add(away)
        result += f"{date} | {home} vs {away} | {league}\n"

    result += "\nTeams with upcoming matches:\n"
    result += ", ".join(sorted(team_names))
    return result

tools = [get_live_scores, get_recent_matches, get_standings, get_upcoming_matches]
agent = create_agent(model=llm, tools=tools, system_prompt="you are a helpful sports assistant")

result = agent.invoke({"messages": [{"role": "user", "content": "List upcoming matches and the team names playing."}]})
print(result["messages"][-1].content)