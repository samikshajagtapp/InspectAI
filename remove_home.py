import re
with open(r'frontend\src\App.jsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove the home view block
start_idx = content.find("if (viewState === 'home') {")
end_idx = content.find("  return (\n    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#f9fafb', overflow: 'hidden' }}>")

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + content[end_idx:]

# Remove viewState from state declarations
content = re.sub(r"const \[viewState, setViewState\] = useState\('home'\);\s*// 'home' \| 'dashboard'\s*\n", "", content)

# Remove setViewState calls
content = re.sub(r"setViewState\('dashboard'\);\s*", "", content)
content = re.sub(r"setViewState\('home'\);\s*", "", content)

# Remove the Back to Home button entirely
# It looks like:
#               {/* Back Arrow button */}
#               <button
#                 onClick={() => {
#                   setReport(null);
#                   setSelectedSample(null);
#                   ...
#                 }}
#                 style={{...}}
#               >
#                 <ArrowLeft size={13} /> Back to Home
#               </button>
# Let's use a regex to remove it
content = re.sub(r"\{\/\* Back Arrow button \*\/\}.*?<ArrowLeft size=\{13\} \/> Back to Home\s*<\/button>\s*", "", content, flags=re.DOTALL)

with open(r'frontend\src\App.jsx', 'w', encoding='utf-8') as f:
    f.write(content)
print("Removed home page and viewState logic.")
