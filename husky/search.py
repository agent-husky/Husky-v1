import json
import serpapi

def table2str(table: list[list[str]]):
    table_str = []
    for row in table:
        row_str = str(row)
        table_str.append(row_str)
    table_str = "\n".join(table_str)
    return table_str

class GoogleSearchAPI:

    def __init__(self, answer_only=False, top_k=1):
        assert(top_k == 1 or not answer_only)
        with open("keys/serpapi_key.json", "r") as f:
            data = json.load(f)
        self.api_key = data["api_key"]
        self.location = data["location"]
        self.answer_only = answer_only
        self.top_k = top_k
    
    def process_scoreboard(self, sports_results):
        teams_info = sports_results["game_spotlight"]["teams"]
        output = []
        for team_info in teams_info:
            if "name" not in team_info or "score" not in team_info:
                return ""
            team = team_info['name']
            team_score_info = team_info['score']
            if isinstance(team_score_info, str):
                output.append(team_score_info)
            else:
                score_list = []
                for quarter, score in team_score_info.items():
                    score_list.append(f"{quarter}: {score}")
                score_str = ", ".join(score_list)
                team_str = f"{team}: {{{score_str}}}"
                output.append(team_str)
        output_str = ", ".join(output)
        return output_str

    def process_result(self, result, use_date=False):
        if not result:
            return
        output = None
        if "sports_results" in result and "game_spotlight" in result["sports_results"]:
            output = self.process_scoreboard(result["sports_results"])
        elif self.top_k == 1 and "answer_box" in result:
            answer_box = result["answer_box"]
            if not self.answer_only and "snippet" in answer_box:
                output = answer_box["snippet"]
            elif "answer" in answer_box:
                output = answer_box["answer"]
        if (not output or isinstance(output, dict)) and "organic_results" in result and len(result["organic_results"]) > 0:
            organic_results = result["organic_results"]
            output = ""
            count = 0
            for organic_result in organic_results:
                if count < self.top_k:
                    if "snippet" in organic_result:
                        title = organic_result["title"]
                        snippet = organic_result["snippet"]
                        out = f"{title}: {snippet}"
                        if use_date and "date" in organic_result:
                            date = organic_result["date"]
                            out = f"(date - {date}) {out}"
                        count += 1
                        if self.top_k == 1:
                            output = out
                        else:
                            out = f"{count}. {out}"
                            output = "\n".join([output, out])
        if not output:  # invalid search results
            return
        output = output.strip()
        return output
    
    def search(self, query: str, use_date=False):
        params = {
            "q": query,
            "location": self.location,
            "api_key": self.api_key
        }
        try:
            search = serpapi.search(params)
            result = search.as_dict()
            output = self.process_result(result, use_date=use_date)
        except:
            output = "Search failed."
        return output
